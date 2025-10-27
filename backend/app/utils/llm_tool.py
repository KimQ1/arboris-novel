# -*- coding: utf-8 -*-
"""OpenAI 兼容型 LLM 工具封装，保持与旧项目一致的接口体验。"""

import os
import logging
import httpx
from dataclasses import asdict, dataclass
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


class LLMClient:
    """异步流式调用封装，兼容 OpenAI SDK。"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("缺少 OPENAI_API_KEY 配置，请在数据库或环境变量中补全。")

        # 配置代理
        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        http_client = None
        if proxy:
            logger.info(f"Using proxy: {proxy}")
            http_client = httpx.AsyncClient(proxy=proxy)

        self._client = AsyncOpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("OPENAI_API_BASE"),
            http_client=http_client
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 120,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, str], None]:
        payload = {
            "model": model or os.environ.get("MODEL", "gpt-3.5-turbo"),
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
            "timeout": timeout,
            **kwargs,
        }
        if response_format:
            payload["response_format"] = {"type": response_format}
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.info(f"Calling LLM API - Base URL: {self._client.base_url}, Model: {payload['model']}")
        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                yield {
                    "content": choice.delta.content,
                    "finish_reason": choice.finish_reason,
                }
        except Exception as e:
            logger.error(f"LLM API Error - Base URL: {self._client.base_url}, Model: {payload['model']}, Error: {str(e)}")
            raise
