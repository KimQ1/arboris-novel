import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import httpx
from fastapi import HTTPException, status
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

from ..core.config import settings
from ..repositories.llm_config_repository import LLMConfigRepository
from ..repositories.system_config_repository import SystemConfigRepository
from ..repositories.user_repository import UserRepository
from ..services.admin_setting_service import AdminSettingService
from ..services.prompt_service import PromptService
from ..services.usage_service import UsageService
from ..utils.llm_tool import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 运行环境未安装时兼容
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:  # pragma: no cover - Ollama 为可选依赖
    OllamaAsyncClient = None


class LLMService:
    """封装与大模型交互的所有逻辑，包括配额控制与配置选择。"""

    # 类变量：用于多 Key 轮询的索引
    _api_key_index = 0

    # 类变量：记录配额耗尽的 API Key 及其失败时间戳
    # 格式: {api_key: timestamp}
    _exhausted_keys: Dict[str, float] = {}

    # 配额耗尽的 Key 黑名单有效期（秒），默认 1 小时
    _exhausted_key_ttl = 3600

    def __init__(self, session):
        self.session = session
        self.llm_repo = LLMConfigRepository(session)
        self.system_config_repo = SystemConfigRepository(session)
        self.user_repo = UserRepository(session)
        self.admin_setting_service = AdminSettingService(session)
        self.usage_service = UsageService(session)
        self._embedding_dimensions: Dict[str, int] = {}

    @classmethod
    def _clean_expired_exhausted_keys(cls) -> None:
        """清理过期的配额耗尽记录"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in cls._exhausted_keys.items()
            if current_time - timestamp > cls._exhausted_key_ttl
        ]
        for key in expired_keys:
            del cls._exhausted_keys[key]
            logger.info("API Key 黑名单已过期，重新启用: %s...", key[:10])

    @classmethod
    def _is_key_exhausted(cls, api_key: str) -> bool:
        """检查 API Key 是否在配额耗尽黑名单中"""
        cls._clean_expired_exhausted_keys()
        return api_key in cls._exhausted_keys

    @classmethod
    def _mark_key_exhausted(cls, api_key: str) -> None:
        """将 API Key 标记为配额耗尽"""
        cls._exhausted_keys[api_key] = time.time()
        logger.warning(
            "API Key 已加入黑名单（%d 小时内不再使用）: %s...",
            cls._exhausted_key_ttl // 3600,
            api_key[:10]
        )

    async def get_llm_response(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        user_id: Optional[int] = None,
        timeout: float = 300.0,
        response_format: Optional[str] = "json_object",
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}, *conversation_history]
        return await self._stream_and_collect(
            messages,
            temperature=temperature,
            user_id=user_id,
            timeout=timeout,
            response_format=response_format,
        )

    async def get_summary(
        self,
        chapter_content: str,
        *,
        temperature: float = 0.2,
        user_id: Optional[int] = None,
        timeout: float = 180.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not system_prompt:
            prompt_service = PromptService(self.session)
            system_prompt = await prompt_service.get_prompt("extraction")
        if not system_prompt:
            logger.error("未配置名为 'extraction' 的摘要提示词，无法生成章节摘要")
            raise HTTPException(status_code=500, detail="未配置摘要提示词，请联系管理员配置 'extraction' 提示词")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chapter_content},
        ]
        return await self._stream_and_collect(messages, temperature=temperature, user_id=user_id, timeout=timeout)

    async def _stream_and_collect(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        user_id: Optional[int],
        timeout: float,
        response_format: Optional[str] = None,
    ) -> str:
        config = await self._resolve_llm_config(user_id)

        # 解析多个 API Key（支持换行符或逗号分隔）
        api_keys = self._parse_api_keys(config["api_key"])

        # 如果只有一个 Key，直接使用旧逻辑
        if len(api_keys) == 1:
            return await self._stream_with_single_key(
                messages=messages,
                api_key=api_keys[0],
                base_url=config.get("base_url"),
                model=config.get("model"),
                temperature=temperature,
                user_id=user_id,
                timeout=timeout,
                response_format=response_format,
            )

        # 多 Key 轮询逻辑
        return await self._stream_with_key_rotation(
            messages=messages,
            api_keys=api_keys,
            base_url=config.get("base_url"),
            model=config.get("model"),
            temperature=temperature,
            user_id=user_id,
            timeout=timeout,
            response_format=response_format,
        )

    def _parse_api_keys(self, api_key_str: Optional[str]) -> List[str]:
        """解析 API Key 字符串，支持换行符或逗号分隔多个 Key"""
        if not api_key_str:
            return []

        # 先按换行符分割，再按逗号分割
        keys = []
        for line in api_key_str.split('\n'):
            for key in line.split(','):
                key = key.strip()
                if key:
                    keys.append(key)

        return keys

    async def _stream_with_key_rotation(
        self,
        messages: List[Dict[str, str]],
        api_keys: List[str],
        base_url: Optional[str],
        model: Optional[str],
        temperature: float,
        user_id: Optional[int],
        timeout: float,
        response_format: Optional[str],
    ) -> str:
        """使用多个 API Key 轮询，遇到 429 错误时自动切换"""
        last_error = None
        failed_keys_count = 0

        # 过滤掉黑名单中的 Key
        available_keys = [key for key in api_keys if not self._is_key_exhausted(key)]
        exhausted_count = len(api_keys) - len(available_keys)

        if exhausted_count > 0:
            logger.info(
                "已过滤 %d 个配额耗尽的 API Key，剩余可用 Key: %d 个",
                exhausted_count,
                len(available_keys)
            )

        if not available_keys:
            logger.error("所有 %d 个 API Key 都已配额耗尽，请稍后重试", len(api_keys))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"所有 {len(api_keys)} 个 API Key 都已达到配额限制，请稍后重试或在设置中添加更多 API Key。"
            )

        # 尝试所有可用的 Key
        for attempt in range(len(available_keys)):
            # 使用轮询索引选择 Key
            key_index = (LLMService._api_key_index + attempt) % len(available_keys)
            current_key = available_keys[key_index]

            logger.info(
                "尝试使用 API Key #%d/%d (可用Key索引: %d)",
                attempt + 1,
                len(available_keys),
                key_index,
            )

            try:
                result = await self._stream_with_single_key(
                    messages=messages,
                    api_key=current_key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    user_id=user_id,
                    timeout=timeout,
                    response_format=response_format,
                )

                # 成功后更新轮询索引到下一个 Key
                LLMService._api_key_index = (key_index + 1) % len(available_keys)
                logger.info("API Key #%d 调用成功，下次将使用 Key #%d", key_index, LLMService._api_key_index)

                return result

            except RateLimitError as exc:
                last_error = exc
                failed_keys_count += 1

                # 将配额耗尽的 Key 加入黑名单
                self._mark_key_exhausted(current_key)

                logger.warning(
                    "API Key #%d 达到配额限制 (429)，已失败 %d/%d 个可用 Key",
                    key_index,
                    failed_keys_count,
                    len(available_keys),
                    exc_info=exc,
                )
                continue
            except HTTPException as exc:
                last_error = exc

                # 503 (服务过载) 和 500 (内部错误) 是临时性错误，尝试下一个 Key
                if exc.status_code in [500, 503]:
                    logger.warning(
                        "API Key #%d 遇到临时错误 (%d)，尝试下一个 Key: %s",
                        key_index,
                        exc.status_code,
                        exc.detail,
                    )
                    continue

                # 其他 HTTP 错误（如 400, 401, 403）直接抛出
                logger.error(
                    "API Key #%d 调用失败（HTTP %d），停止轮询: %s",
                    key_index,
                    exc.status_code,
                    exc.detail,
                    exc_info=exc,
                )
                raise
            except Exception as exc:
                # 其他未知错误直接抛出
                logger.error(
                    "API Key #%d 调用失败（未知错误），停止轮询: %s",
                    key_index,
                    str(exc),
                    exc_info=exc,
                )
                raise

        # 所有可用 Key 都失败了
        logger.error("所有 %d 个可用 API Key 都已达到配额限制，请求失败", len(available_keys))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"所有可用的 API Key 都已达到配额限制，请稍后重试或在设置中添加更多 API Key。建议等待配额重置后再试。"
        )

    async def _stream_with_single_key(
        self,
        messages: List[Dict[str, str]],
        api_key: str,
        base_url: Optional[str],
        model: Optional[str],
        temperature: float,
        user_id: Optional[int],
        timeout: float,
        response_format: Optional[str],
    ) -> str:
        """使用单个 API Key 进行流式调用"""
        client = LLMClient(api_key=api_key, base_url=base_url)

        chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages]

        full_response = ""
        finish_reason = None

        logger.info(
            "Streaming LLM response: model=%s user_id=%s messages=%d",
            model,
            user_id,
            len(messages),
        )

        try:
            async for part in client.stream_chat(
                messages=chat_messages,
                model=model,
                temperature=temperature,
                timeout=int(timeout),
                response_format=response_format,
            ):
                if part.get("content"):
                    full_response += part["content"]
                if part.get("finish_reason"):
                    finish_reason = part["finish_reason"]
        except RateLimitError as exc:
            # 429 错误，向上抛出以便轮询逻辑处理
            raise
        except InternalServerError as exc:
            detail = "AI 服务内部错误，请稍后重试"
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    payload = response.json()
                    error_data = payload.get("error", {}) if isinstance(payload, dict) else {}
                    detail = error_data.get("message_zh") or error_data.get("message") or detail
                except Exception:
                    detail = str(exc) or detail
            else:
                detail = str(exc) or detail
            logger.error(
                "LLM stream internal error: model=%s user_id=%s detail=%s",
                model,
                user_id,
                detail,
                exc_info=exc,
            )
            raise HTTPException(status_code=503, detail=detail)
        except (httpx.RemoteProtocolError, httpx.ReadTimeout, APIConnectionError, APITimeoutError) as exc:
            if isinstance(exc, httpx.RemoteProtocolError):
                detail = "AI 服务连接被意外中断，请稍后重试"
            elif isinstance(exc, (httpx.ReadTimeout, APITimeoutError)):
                detail = "AI 服务响应超时，请稍后重试"
            else:
                detail = "无法连接到 AI 服务，请稍后重试"
            logger.error(
                "LLM stream failed: model=%s user_id=%s detail=%s",
                model,
                user_id,
                detail,
                exc_info=exc,
            )
            raise HTTPException(status_code=503, detail=detail) from exc

        logger.debug(
            "LLM response collected: model=%s user_id=%s finish_reason=%s preview=%s",
            model,
            user_id,
            finish_reason,
            full_response[:500],
        )

        if finish_reason == "length":
            logger.warning(
                "LLM response truncated: model=%s user_id=%s response_length=%d",
                model,
                user_id,
                len(full_response),
            )
            raise HTTPException(
                status_code=500,
                detail=f"AI 响应因长度限制被截断（已生成 {len(full_response)} 字符），请缩短输入内容或调整模型参数"
            )

        if not full_response:
            logger.error(
                "LLM returned empty response: model=%s user_id=%s finish_reason=%s",
                model,
                user_id,
                finish_reason,
            )
            raise HTTPException(
                status_code=500,
                detail=f"AI 未返回有效内容（结束原因: {finish_reason or '未知'}），请稍后重试或联系管理员"
            )

        await self.usage_service.increment("api_request_count")
        logger.info(
            "LLM response success: model=%s user_id=%s chars=%d",
            model,
            user_id,
            len(full_response),
        )
        return full_response

    async def _resolve_llm_config(self, user_id: Optional[int]) -> Dict[str, Optional[str]]:
        if user_id:
            config = await self.llm_repo.get_by_user(user_id)
            if config and config.llm_provider_api_key:
                return {
                    "api_key": config.llm_provider_api_key,
                    "base_url": config.llm_provider_url,
                    "model": config.llm_provider_model,
                }

        # 检查每日使用次数限制
        if user_id:
            await self._enforce_daily_limit(user_id)

        api_key = await self._get_config_value("llm.api_key")
        base_url = await self._get_config_value("llm.base_url")
        model = await self._get_config_value("llm.model")

        if not api_key:
            logger.error("未配置默认 LLM API Key，且用户 %s 未设置自定义 API Key", user_id)
            raise HTTPException(
                status_code=500,
                detail="未配置默认 LLM API Key，请联系管理员配置系统默认 API Key 或在个人设置中配置自定义 API Key"
            )

        return {"api_key": api_key, "base_url": base_url, "model": model}

    async def get_embedding(
        self,
        text: str,
        *,
        user_id: Optional[int] = None,
        model: Optional[str] = None,
    ) -> List[float]:
        """生成文本向量，用于章节 RAG 检索，支持 openai 与 ollama 双提供方。"""
        provider = await self._get_config_value("embedding.provider") or "openai"
        default_model = (
            await self._get_config_value("ollama.embedding_model") or "nomic-embed-text:latest"
            if provider == "ollama"
            else await self._get_config_value("embedding.model") or "text-embedding-3-large"
        )
        target_model = model or default_model

        if provider == "ollama":
            if OllamaAsyncClient is None:
                logger.error("未安装 ollama 依赖，无法调用本地嵌入模型。")
                raise HTTPException(status_code=500, detail="缺少 Ollama 依赖，请先安装 ollama 包。")

            base_url = (
                await self._get_config_value("ollama.embedding_base_url")
                or await self._get_config_value("embedding.base_url")
            )
            client = OllamaAsyncClient(host=base_url)
            try:
                response = await client.embeddings(model=target_model, prompt=text)
            except Exception as exc:  # pragma: no cover - 本地服务调用失败
                logger.error(
                    "Ollama 嵌入请求失败: model=%s base_url=%s error=%s",
                    target_model,
                    base_url,
                    exc,
                    exc_info=True,
                )
                return []
            embedding: Optional[List[float]]
            if isinstance(response, dict):
                embedding = response.get("embedding")
            else:
                embedding = getattr(response, "embedding", None)
            if not embedding:
                logger.warning("Ollama 返回空向量: model=%s", target_model)
                return []
            if not isinstance(embedding, list):
                embedding = list(embedding)
        else:
            # OpenAI 兼容接口，使用 Key 轮换机制
            config = await self._resolve_llm_config(user_id)
            embedding_api_key = await self._get_config_value("embedding.api_key")
            base_url = await self._get_config_value("embedding.base_url") or config.get("base_url")

            # 解析 API Keys（支持多个 Key）
            if embedding_api_key:
                api_keys = self._parse_api_keys(embedding_api_key)
            else:
                # 回退使用 LLM 的 API Keys
                api_keys = self._parse_api_keys(config["api_key"])

            if not api_keys:
                logger.error("未配置嵌入模型 API Key")
                return []

            # 使用 Key 轮换机制
            embedding = await self._get_embedding_with_key_rotation(
                text=text,
                api_keys=api_keys,
                base_url=base_url,
                model=target_model,
                user_id=user_id,
            )

            if not embedding:
                return []

        if not isinstance(embedding, list):
            embedding = list(embedding)

        dimension = len(embedding)
        if not dimension:
            vector_size_str = await self._get_config_value("embedding.model_vector_size")
            if vector_size_str:
                dimension = int(vector_size_str)
        if dimension:
            self._embedding_dimensions[target_model] = dimension
        return embedding

    async def _get_embedding_with_key_rotation(
        self,
        text: str,
        api_keys: List[str],
        base_url: Optional[str],
        model: str,
        user_id: Optional[int],
    ) -> List[float]:
        """使用多个 API Key 轮询获取嵌入向量"""
        last_error = None
        failed_keys_count = 0

        # 过滤掉黑名单中的 Key
        available_keys = [key for key in api_keys if not self._is_key_exhausted(key)]
        exhausted_count = len(api_keys) - len(available_keys)

        if exhausted_count > 0:
            logger.info(
                "[嵌入] 已过滤 %d 个配额耗尽的 API Key，剩余可用 Key: %d 个",
                exhausted_count,
                len(available_keys)
            )

        if not available_keys:
            logger.error("[嵌入] 所有 %d 个 API Key 都已配额耗尽", len(api_keys))
            return []

        # 尝试所有可用的 Key
        for attempt in range(len(available_keys)):
            # 使用轮询索引选择 Key
            key_index = (LLMService._api_key_index + attempt) % len(available_keys)
            current_key = available_keys[key_index]

            logger.info(
                "[嵌入] 尝试使用 API Key #%d/%d (可用Key索引: %d)",
                attempt + 1,
                len(available_keys),
                key_index,
            )

            client = AsyncOpenAI(api_key=current_key, base_url=base_url)

            try:
                response = await client.embeddings.create(
                    input=text,
                    model=model,
                )

                if not response.data:
                    logger.warning("[嵌入] API 返回空数据: model=%s", model)
                    continue

                # 成功后更新轮询索引到下一个 Key
                LLMService._api_key_index = (key_index + 1) % len(available_keys)
                logger.info("[嵌入] API Key #%d 调用成功", key_index)

                return response.data[0].embedding

            except RateLimitError as exc:
                last_error = exc
                failed_keys_count += 1

                # 将配额耗尽的 Key 加入黑名单
                self._mark_key_exhausted(current_key)

                logger.warning(
                    "[嵌入] API Key #%d 达到配额限制 (429)，已失败 %d/%d 个可用 Key",
                    key_index,
                    failed_keys_count,
                    len(available_keys),
                    exc_info=exc,
                )
                continue

            except Exception as exc:
                # 其他错误（如 invalid API key）记录但继续尝试下一个 Key
                logger.error(
                    "[嵌入] API Key #%d 调用失败: model=%s base_url=%s error=%s",
                    key_index,
                    model,
                    base_url,
                    exc,
                    exc_info=True,
                )
                # 对于非配额错误，也尝试下一个 Key
                continue

        # 所有 Key 都失败了
        logger.error("[嵌入] 所有 %d 个可用 API Key 都已失败", len(available_keys))
        return []

    async def get_embedding_dimension(self, model: Optional[str] = None) -> Optional[int]:
        """获取嵌入向量维度，优先返回缓存结果，其次读取配置。"""
        provider = await self._get_config_value("embedding.provider") or "openai"
        default_model = (
            await self._get_config_value("ollama.embedding_model") or "nomic-embed-text:latest"
            if provider == "ollama"
            else await self._get_config_value("embedding.model") or "text-embedding-3-large"
        )
        target_model = model or default_model
        if target_model in self._embedding_dimensions:
            return self._embedding_dimensions[target_model]
        vector_size_str = await self._get_config_value("embedding.model_vector_size")
        return int(vector_size_str) if vector_size_str else None

    async def _enforce_daily_limit(self, user_id: int) -> None:
        limit_str = await self.admin_setting_service.get("daily_request_limit", "100")
        limit = int(limit_str or 10)
        used = await self.user_repo.get_daily_request(user_id)
        if used >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="今日请求次数已达上限，请明日再试或设置自定义 API Key。",
            )
        await self.user_repo.increment_daily_request(user_id)
        await self.session.commit()

    async def _get_config_value(self, key: str) -> Optional[str]:
        # 环境变量优先级高于数据库配置
        # 这样可以让本地开发和 Docker 部署使用不同的配置，而不会互相冲突
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value:
            return env_value

        # 如果没有环境变量，则从数据库读取
        record = await self.system_config_repo.get_by_key(key)
        if record:
            return record.value

        return None
