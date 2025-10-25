import json
import logging
import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import settings
from ...core.dependencies import get_current_user
from ...db.session import get_session
from ...models.novel import Chapter, ChapterOutline
from ...schemas.novel import (
    DeleteChapterRequest,
    EditChapterRequest,
    EvaluateChapterRequest,
    GenerateChapterRequest,
    GenerateOutlineRequest,
    NovelProject as NovelProjectSchema,
    RegenerateEmbeddingsRequest,
    SelectVersionRequest,
    UpdateChapterOutlineRequest,
)
from ...schemas.user import UserInDB
from ...services.chapter_context_service import ChapterContextService
from ...services.chapter_ingest_service import ChapterIngestionService
from ...services.llm_service import LLMService
from ...services.novel_service import NovelService
from ...services.prompt_service import PromptService
from ...services.vector_store_service import VectorStoreService
from ...utils.json_utils import remove_think_tags, unwrap_markdown_json
from ...repositories.system_config_repository import SystemConfigRepository

router = APIRouter(prefix="/api/writer", tags=["Writer"])
logger = logging.getLogger(__name__)


async def _load_project_schema(service: NovelService, project_id: str, user_id: int) -> NovelProjectSchema:
    return await service.get_project_schema(project_id, user_id)


def _extract_tail_excerpt(text: Optional[str], limit: int = 500) -> str:
    """截取章节结尾文本，默认保留 500 字。"""
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


@router.post("/novels/{project_id}/chapters/generate", response_model=NovelProjectSchema)
async def generate_chapter(
    project_id: str,
    request: GenerateChapterRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    novel_service = NovelService(session)
    prompt_service = PromptService(session)
    llm_service = LLMService(session)

    project = await novel_service.ensure_project_owner(project_id, current_user.id)
    logger.info("用户 %s 开始为项目 %s 生成第 %s 章", current_user.id, project_id, request.chapter_number)
    outline = await novel_service.get_outline(project_id, request.chapter_number)
    if not outline:
        logger.warning("项目 %s 未找到第 %s 章纲要，生成流程终止", project_id, request.chapter_number)
        raise HTTPException(status_code=404, detail="蓝图中未找到对应章节纲要")

    chapter = await novel_service.get_or_create_chapter(project_id, request.chapter_number)
    chapter.real_summary = None
    chapter.selected_version_id = None
    chapter.status = "generating"
    await session.commit()

    # 使用 try-except 包裹整个生成流程，确保异常时更新章节状态
    try:
        outlines_map = {item.chapter_number: item for item in project.outlines}
        # 收集所有可用的历史章节摘要，便于在 Prompt 中提供前情背景
        completed_chapters = []
        latest_prev_number = -1
        previous_summary_text = ""
        previous_tail_excerpt = ""
        for existing in project.chapters:
            if existing.chapter_number >= request.chapter_number:
                continue
            if existing.selected_version is None or not existing.selected_version.content:
                continue
            if not existing.real_summary:
                summary = await llm_service.get_summary(
                    existing.selected_version.content,
                    temperature=0.15,
                    user_id=current_user.id,
                    timeout=180.0,
                )
                existing.real_summary = remove_think_tags(summary)
                await session.commit()
            completed_chapters.append(
                {
                    "chapter_number": existing.chapter_number,
                    "title": outlines_map.get(existing.chapter_number).title if outlines_map.get(existing.chapter_number) else f"第{existing.chapter_number}章",
                    "summary": existing.real_summary,
                }
            )
            if existing.chapter_number > latest_prev_number:
                latest_prev_number = existing.chapter_number
                previous_summary_text = existing.real_summary or ""
                previous_tail_excerpt = _extract_tail_excerpt(existing.selected_version.content)

        project_schema = await novel_service._serialize_project(project)
        blueprint_dict = project_schema.blueprint.model_dump()

        if "relationships" in blueprint_dict and blueprint_dict["relationships"]:
            for relation in blueprint_dict["relationships"]:
                if "character_from" in relation:
                    relation["from"] = relation.pop("character_from")
                if "character_to" in relation:
                    relation["to"] = relation.pop("character_to")

        # 蓝图中禁止携带章节级别的细节信息，避免重复传输大段场景或对话内容
        banned_blueprint_keys = {
            "chapter_outline",
            "chapter_summaries",
            "chapter_details",
            "chapter_dialogues",
            "chapter_events",
            "conversation_history",
            "character_timelines",
        }
        for key in banned_blueprint_keys:
            if key in blueprint_dict:
                blueprint_dict.pop(key, None)

        writer_prompt = await prompt_service.get_prompt("writing")
        if not writer_prompt:
            logger.error("未配置名为 'writing' 的写作提示词，无法生成章节内容")
            raise HTTPException(status_code=500, detail="缺少写作提示词，请联系管理员配置 'writing' 提示词")

        # 初始化向量检索服务，若未配置则自动降级为纯提示词生成
        vector_store: Optional[VectorStoreService]
        if not settings.vector_store_enabled:
            vector_store = None
        else:
            try:
                vector_store = VectorStoreService()
            except RuntimeError as exc:
                logger.warning("向量库初始化失败，RAG 检索被禁用: %s", exc)
                vector_store = None
        context_service = ChapterContextService(llm_service=llm_service, vector_store=vector_store)

        outline_title = outline.title or f"第{outline.chapter_number}章"
        outline_summary = outline.summary or "暂无摘要"
        query_parts = [outline_title, outline_summary]
        if request.writing_notes:
            query_parts.append(request.writing_notes)
        rag_query = "\n".join(part for part in query_parts if part)
        rag_context = await context_service.retrieve_for_generation(
            project_id=project_id,
            query_text=rag_query or outline.title or outline.summary or "",
            user_id=current_user.id,
        )
        chunk_count = len(rag_context.chunks) if rag_context and rag_context.chunks else 0
        summary_count = len(rag_context.summaries) if rag_context and rag_context.summaries else 0
        logger.info(
            "项目 %s 第 %s 章检索到 %s 个剧情片段和 %s 条摘要",
            project_id,
            request.chapter_number,
            chunk_count,
            summary_count,
        )
        # print("rag_context:",rag_context)
        # 将蓝图、前情、RAG 检索结果拼装成结构化段落，供模型理解
        blueprint_text = json.dumps(blueprint_dict, ensure_ascii=False, indent=2)
        completed_lines = [
            f"- 第{item['chapter_number']}章 - {item['title']}:{item['summary']}"
            for item in completed_chapters
        ]
        previous_summary_text = previous_summary_text or "暂无可用摘要"
        previous_tail_excerpt = previous_tail_excerpt or "暂无上一章结尾内容"
        completed_section = "\n".join(completed_lines) if completed_lines else "暂无前情摘要"
        rag_chunks_text = "\n\n".join(rag_context.chunk_texts()) if rag_context.chunks else "未检索到章节片段"
        rag_summaries_text = "\n".join(rag_context.summary_lines()) if rag_context.summaries else "未检索到章节摘要"
        writing_notes = request.writing_notes or "无额外写作指令"

        prompt_sections = [
            ("[世界蓝图](JSON)", blueprint_text),
            # ("[前情摘要]", completed_section),
            ("[上一章摘要]", previous_summary_text),
            ("[上一章结尾]", previous_tail_excerpt),
            ("[检索到的剧情上下文](Markdown)", rag_chunks_text),
            ("[检索到的章节摘要]", rag_summaries_text),
            (
                "[当前章节目标]",
                f"标题：{outline_title}\n摘要：{outline_summary}\n写作要求：{writing_notes}",
            ),
        ]
        prompt_input = "\n\n".join(f"{title}\n{content}" for title, content in prompt_sections if content)
        logger.debug("章节写作提示词：%s\n%s", writer_prompt, prompt_input)
        async def _generate_single_version(idx: int) -> Dict:
            try:
                response = await llm_service.get_llm_response(
                    system_prompt=writer_prompt,
                    conversation_history=[{"role": "user", "content": prompt_input}],
                    temperature=0.9,
                    user_id=current_user.id,
                    timeout=600.0,
                )
                cleaned = remove_think_tags(response)
                normalized = unwrap_markdown_json(cleaned)
                try:
                    return json.loads(normalized)
                except json.JSONDecodeError as parse_err:
                    logger.warning(
                        "项目 %s 第 %s 章第 %s 个版本 JSON 解析失败，将原始内容作为纯文本处理: %s",
                        project_id,
                        request.chapter_number,
                        idx + 1,
                        parse_err,
                    )
                    return {"content": normalized}
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception(
                    "项目 %s 生成第 %s 章第 %s 个版本时发生异常: %s",
                    project_id,
                    request.chapter_number,
                    idx + 1,
                    exc,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"生成章节第 {idx + 1} 个版本时失败: {str(exc)[:200]}"
                )

        version_count = await _resolve_version_count(session)
        logger.info(
            "项目 %s 第 %s 章计划生成 %s 个版本",
            project_id,
            request.chapter_number,
            version_count,
        )
        raw_versions = []
        for idx in range(version_count):
            raw_versions.append(await _generate_single_version(idx))

        # 直接传递原始版本数据和元数据给 replace_chapter_versions
        # 让 _normalize_version_content 函数处理所有内容提取逻辑
        # 这样可以支持 content, chapter_content, full_content 等多种字段
        contents: List[str] = []
        metadata: List[Dict] = []
        for variant in raw_versions:
            if isinstance(variant, dict):
                # 将原始 dict 作为 content，让 _normalize_version_content 提取
                contents.append(variant)
                metadata.append(variant)
            else:
                contents.append(str(variant))
                metadata.append({"raw": variant})

        await novel_service.replace_chapter_versions(chapter, contents, metadata)
        logger.info(
            "项目 %s 第 %s 章生成完成，已写入 %s 个版本",
            project_id,
            request.chapter_number,
            len(contents),
        )
        return await _load_project_schema(novel_service, project_id, current_user.id)
    except HTTPException:
        # HTTPException 需要向上抛出，但先更新章节状态
        chapter.status = "failed"
        await session.commit()
        logger.error(
            "项目 %s 第 %s 章生成失败（HTTPException），状态已更新为 failed",
            project_id,
            request.chapter_number,
        )
        raise
    except Exception as exc:
        # 其他异常也需要更新章节状态
        chapter.status = "failed"
        await session.commit()
        logger.exception(
            "项目 %s 第 %s 章生成失败（未知异常）: %s",
            project_id,
            request.chapter_number,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"生成章节失败: {str(exc)[:200]}"
        )


async def _resolve_version_count(session: AsyncSession) -> int:
    repo = SystemConfigRepository(session)
    record = await repo.get_by_key("writer.chapter_versions")
    if record:
        try:
            value = int(record.value)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    env_value = os.getenv("WRITER_CHAPTER_VERSION_COUNT")
    if env_value:
        try:
            value = int(env_value)
            if value > 0:
                return value
        except ValueError:
            pass
    return 3


@router.post("/novels/{project_id}/chapters/select", response_model=NovelProjectSchema)
async def select_chapter_version(
    project_id: str,
    request: SelectVersionRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    novel_service = NovelService(session)
    llm_service = LLMService(session)

    project = await novel_service.ensure_project_owner(project_id, current_user.id)
    chapter = next((ch for ch in project.chapters if ch.chapter_number == request.chapter_number), None)
    if not chapter:
        logger.warning("项目 %s 未找到第 %s 章，无法选择版本", project_id, request.chapter_number)
        raise HTTPException(status_code=404, detail="章节不存在")

    selected = await novel_service.select_chapter_version(chapter, request.version_index)
    logger.info(
        "用户 %s 选择了项目 %s 第 %s 章的第 %s 个版本",
        current_user.id,
        project_id,
        request.chapter_number,
        request.version_index,
    )
    if selected and selected.content:
        summary = await llm_service.get_summary(
            selected.content,
            temperature=0.15,
            user_id=current_user.id,
            timeout=180.0,
        )
        chapter.real_summary = remove_think_tags(summary)
        await session.commit()

        # 选定版本后同步向量库，确保后续章节可检索到最新内容
        vector_store: Optional[VectorStoreService]
        if not settings.vector_store_enabled:
            vector_store = None
        else:
            try:
                vector_store = VectorStoreService()
            except RuntimeError as exc:
                logger.warning("向量库初始化失败，跳过章节向量同步: %s", exc)
                vector_store = None

        if vector_store:
            ingestion_service = ChapterIngestionService(llm_service=llm_service, vector_store=vector_store)
            outline = next((item for item in project.outlines if item.chapter_number == chapter.chapter_number), None)
            chapter_title = outline.title if outline and outline.title else f"第{chapter.chapter_number}章"
            await ingestion_service.ingest_chapter(
                project_id=project_id,
                chapter_number=chapter.chapter_number,
                title=chapter_title,
                content=selected.content,
                summary=chapter.real_summary,
                user_id=current_user.id,
            )
            logger.info(
                "项目 %s 第 %s 章已同步至向量库",
                project_id,
                chapter.chapter_number,
            )

    return await _load_project_schema(novel_service, project_id, current_user.id)


@router.post("/novels/{project_id}/chapters/evaluate", response_model=NovelProjectSchema)
async def evaluate_chapter(
    project_id: str,
    request: EvaluateChapterRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    novel_service = NovelService(session)
    prompt_service = PromptService(session)
    llm_service = LLMService(session)

    project = await novel_service.ensure_project_owner(project_id, current_user.id)
    chapter = next((ch for ch in project.chapters if ch.chapter_number == request.chapter_number), None)
    if not chapter:
        logger.warning("项目 %s 未找到第 %s 章，无法执行评估", project_id, request.chapter_number)
        raise HTTPException(status_code=404, detail="章节不存在")
    if not chapter.versions:
        logger.warning("项目 %s 第 %s 章无可评估版本", project_id, request.chapter_number)
        raise HTTPException(status_code=400, detail="无可评估的章节版本")

    # 防止重复评审：如果正在评审中，拒绝新的评审请求
    if chapter.status == "evaluating":
        logger.warning("项目 %s 第 %s 章正在评审中，拒绝重复评审请求", project_id, request.chapter_number)
        raise HTTPException(status_code=409, detail="章节正在评审中，请稍后再试")

    # 标记为评审中状态
    chapter.status = "evaluating"
    await session.commit()

    evaluator_prompt = await prompt_service.get_prompt("evaluation")
    if not evaluator_prompt:
        logger.error("缺少评估提示词，项目 %s 第 %s 章评估失败", project_id, request.chapter_number)
        raise HTTPException(status_code=500, detail="缺少评估提示词，请联系管理员配置 'evaluation' 提示词")

    project_schema = await novel_service._serialize_project(project)
    blueprint_dict = project_schema.blueprint.model_dump()

    versions_to_evaluate = [
        {"version_id": idx + 1, "content": version.content}
        for idx, version in enumerate(sorted(chapter.versions, key=lambda item: item.created_at))
    ]
    # print("blueprint_dict:",blueprint_dict)
    evaluator_payload = {
        "novel_blueprint": blueprint_dict,
        "content_to_evaluate": {
            "chapter_number": chapter.chapter_number,
            "versions": versions_to_evaluate,
        },
    }

    try:
        evaluation_raw = await llm_service.get_llm_response(
            system_prompt=evaluator_prompt,
            conversation_history=[{"role": "user", "content": json.dumps(evaluator_payload, ensure_ascii=False)}],
            temperature=0.3,
            user_id=current_user.id,
            timeout=360.0,
        )
        evaluation_clean = remove_think_tags(evaluation_raw)
        await novel_service.add_chapter_evaluation(chapter, None, evaluation_clean)
        logger.info("项目 %s 第 %s 章评估完成", project_id, request.chapter_number)
    except Exception as exc:
        # 评审失败时恢复状态
        chapter.status = "evaluation_failed"
        await session.commit()
        logger.error("项目 %s 第 %s 章评估失败: %s", project_id, request.chapter_number, exc)
        raise

    return await _load_project_schema(novel_service, project_id, current_user.id)


@router.post("/novels/{project_id}/chapters/outline", response_model=NovelProjectSchema)
async def generate_chapter_outline(
    project_id: str,
    request: GenerateOutlineRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    novel_service = NovelService(session)
    prompt_service = PromptService(session)
    llm_service = LLMService(session)

    await novel_service.ensure_project_owner(project_id, current_user.id)
    logger.info(
        "用户 %s 请求生成项目 %s 的章节大纲，起始章节 %s，数量 %s",
        current_user.id,
        project_id,
        request.start_chapter,
        request.num_chapters,
    )
    outline_prompt = await prompt_service.get_prompt("outline")
    if not outline_prompt:
        logger.error("缺少大纲提示词，项目 %s 大纲生成失败", project_id)
        raise HTTPException(status_code=500, detail="缺少大纲提示词，请联系管理员配置 'outline' 提示词")

    project_schema = await novel_service.get_project_schema(project_id, current_user.id)
    blueprint_dict = project_schema.blueprint.model_dump()

    # 获取起始章节之前的所有已完成章节的摘要信息
    completed_chapters = []
    if request.start_chapter > 1:
        for chapter in project_schema.chapters:
            if chapter.chapter_number < request.start_chapter and chapter.content:
                # 只传递章节号、标题和摘要，不传递完整内容
                completed_chapters.append({
                    "chapter_number": chapter.chapter_number,
                    "title": chapter.title,
                    "summary": chapter.real_summary or chapter.summary,
                })
        logger.info(
            "项目 %s 找到 %s 个已完成章节，将用于调整后续大纲",
            project_id,
            len(completed_chapters)
        )

    payload = {
        "novel_blueprint": blueprint_dict,
        "completed_chapters": completed_chapters,  # 添加已完成章节信息
        "wait_to_generate": {
            "start_chapter": request.start_chapter,
            "num_chapters": request.num_chapters,
        },
    }

    response = await llm_service.get_llm_response(
        system_prompt=outline_prompt,
        conversation_history=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.7,
        user_id=current_user.id,
        timeout=360.0,
    )
    normalized = unwrap_markdown_json(remove_think_tags(response))
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError as exc:
        logger.error(
            "项目 %s 大纲生成 JSON 解析失败: %s, 原始内容预览: %s",
            project_id,
            exc,
            normalized[:500],
        )
        raise HTTPException(
            status_code=500,
            detail=f"章节大纲生成失败，AI 返回的内容格式不正确: {str(exc)}"
        ) from exc

    new_outlines = data.get("chapters", [])
    for item in new_outlines:
        stmt = (
            select(ChapterOutline)
            .where(
                ChapterOutline.project_id == project_id,
                ChapterOutline.chapter_number == item.get("chapter_number"),
            )
        )
        result = await session.execute(stmt)
        record = result.scalars().first()
        if record:
            record.title = item.get("title", record.title)
            record.summary = item.get("summary", record.summary)
        else:
            session.add(
                ChapterOutline(
                    project_id=project_id,
                    chapter_number=item.get("chapter_number"),
                    title=item.get("title", ""),
                    summary=item.get("summary"),
                )
            )
    await session.commit()
    logger.info("项目 %s 章节大纲生成完成", project_id)

    return await novel_service.get_project_schema(project_id, current_user.id)


@router.post("/novels/{project_id}/chapters/update-outline", response_model=NovelProjectSchema)
async def update_chapter_outline(
    project_id: str,
    request: UpdateChapterOutlineRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    novel_service = NovelService(session)
    await novel_service.ensure_project_owner(project_id, current_user.id)
    logger.info(
        "用户 %s 更新项目 %s 第 %s 章大纲",
        current_user.id,
        project_id,
        request.chapter_number,
    )

    stmt = (
        select(ChapterOutline)
        .where(
            ChapterOutline.project_id == project_id,
            ChapterOutline.chapter_number == request.chapter_number,
        )
    )
    result = await session.execute(stmt)
    outline = result.scalars().first()
    if not outline:
        outline = ChapterOutline(
            project_id=project_id,
            chapter_number=request.chapter_number,
        )
        session.add(outline)

    outline.title = request.title
    outline.summary = request.summary
    await session.commit()
    logger.info("项目 %s 第 %s 章大纲已更新", project_id, request.chapter_number)

    return await novel_service.get_project_schema(project_id, current_user.id)


@router.post("/novels/{project_id}/chapters/delete", response_model=NovelProjectSchema)
async def delete_chapters(
    project_id: str,
    request: DeleteChapterRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    if not request.chapter_numbers:
        logger.warning("项目 %s 删除章节时未提供章节号", project_id)
        raise HTTPException(status_code=400, detail="请提供要删除的章节号列表")
    novel_service = NovelService(session)
    llm_service = LLMService(session)
    await novel_service.ensure_project_owner(project_id, current_user.id)
    logger.info(
        "用户 %s 删除项目 %s 的章节 %s",
        current_user.id,
        project_id,
        request.chapter_numbers,
    )
    await novel_service.delete_chapters(project_id, request.chapter_numbers)

    # 删除章节时同步清理向量库，避免过时内容被检索
    vector_store: Optional[VectorStoreService]
    if not settings.vector_store_enabled:
        vector_store = None
    else:
        try:
            vector_store = VectorStoreService()
        except RuntimeError as exc:
            logger.warning("向量库初始化失败，跳过章节向量删除: %s", exc)
            vector_store = None

    if vector_store:
        ingestion_service = ChapterIngestionService(llm_service=llm_service, vector_store=vector_store)
        await ingestion_service.delete_chapters(project_id, request.chapter_numbers)
        logger.info(
            "项目 %s 已从向量库移除章节 %s",
            project_id,
            request.chapter_numbers,
        )

    return await novel_service.get_project_schema(project_id, current_user.id)


@router.post("/novels/{project_id}/chapters/edit", response_model=NovelProjectSchema)
async def edit_chapter(
    project_id: str,
    request: EditChapterRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    novel_service = NovelService(session)
    llm_service = LLMService(session)

    project = await novel_service.ensure_project_owner(project_id, current_user.id)
    chapter = next((ch for ch in project.chapters if ch.chapter_number == request.chapter_number), None)
    if not chapter or chapter.selected_version is None:
        logger.warning("项目 %s 第 %s 章尚未生成或未选择版本，无法编辑", project_id, request.chapter_number)
        raise HTTPException(status_code=404, detail="章节尚未生成或未选择版本")

    chapter.selected_version.content = request.content
    chapter.word_count = len(request.content)
    logger.info("用户 %s 更新了项目 %s 第 %s 章内容", current_user.id, project_id, request.chapter_number)

    if request.content.strip():
        summary = await llm_service.get_summary(
            request.content,
            temperature=0.15,
            user_id=current_user.id,
            timeout=180.0,
        )
        chapter.real_summary = remove_think_tags(summary)
    await session.commit()

    vector_store: Optional[VectorStoreService]
    if not settings.vector_store_enabled:
        vector_store = None
    else:
        try:
            vector_store = VectorStoreService()
        except RuntimeError as exc:
            logger.warning("向量库初始化失败，跳过章节向量更新: %s", exc)
            vector_store = None

    if vector_store and chapter.selected_version and chapter.selected_version.content:
        ingestion_service = ChapterIngestionService(llm_service=llm_service, vector_store=vector_store)
        outline = next((item for item in project.outlines if item.chapter_number == chapter.chapter_number), None)
        chapter_title = outline.title if outline and outline.title else f"第{chapter.chapter_number}章"
        await ingestion_service.ingest_chapter(
            project_id=project_id,
            chapter_number=chapter.chapter_number,
            title=chapter_title,
            content=chapter.selected_version.content,
            summary=chapter.real_summary,
            user_id=current_user.id,
        )
        logger.info("项目 %s 第 %s 章更新内容已同步至向量库", project_id, chapter.chapter_number)

    return await novel_service.get_project_schema(project_id, current_user.id)


@router.post("/novels/{project_id}/chapters/regenerate-embeddings", response_model=NovelProjectSchema)
async def regenerate_embeddings(
    project_id: str,
    request: RegenerateEmbeddingsRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserInDB = Depends(get_current_user),
) -> NovelProjectSchema:
    """重新生成章节的嵌入向量，用于修复之前失败的向量生成。"""
    if not settings.vector_store_enabled:
        raise HTTPException(status_code=400, detail="向量库未启用")

    novel_service = NovelService(session)
    llm_service = LLMService(session)

    project = await novel_service.ensure_project_owner(project_id, current_user.id)

    try:
        vector_store = VectorStoreService()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"向量库初始化失败: {exc}")

    ingestion_service = ChapterIngestionService(llm_service=llm_service, vector_store=vector_store)

    # 确定要处理的章节
    if request.chapter_numbers:
        # 处理指定的章节（优先使用已选择版本，如果没有则使用第一个版本）
        chapters_to_process = []
        for chapter_number in request.chapter_numbers:
            chapter = next((c for c in project.chapters if c.chapter_number == chapter_number), None)
            if not chapter:
                logger.warning("章节 %s 不存在，跳过", chapter_number)
                continue

            # 优先使用已选择版本，否则使用第一个版本
            if chapter.selected_version:
                chapters_to_process.append(chapter)
            elif chapter.versions and len(chapter.versions) > 0:
                # 临时设置 selected_version 为第一个版本
                chapter.selected_version = chapter.versions[0]
                chapters_to_process.append(chapter)
                logger.info("章节 %s 未选择版本，将使用第一个版本生成嵌入", chapter_number)
            else:
                logger.warning("章节 %s 没有任何版本内容，跳过", chapter_number)

        if not chapters_to_process:
            raise HTTPException(status_code=404, detail="指定的章节都不存在或没有内容")
    else:
        # 处理所有已选择版本的章节
        chapters_to_process = [c for c in project.chapters if c.selected_version]
        if not chapters_to_process:
            raise HTTPException(status_code=404, detail="项目中没有已选择版本的章节")

    logger.info(
        "用户 %s 开始重新生成嵌入向量: project=%s chapters=%s",
        current_user.id,
        project_id,
        [c.chapter_number for c in chapters_to_process],
    )

    # 重新生成每个章节的向量
    success_count = 0
    failed_chapters = []

    for chapter in chapters_to_process:
        try:
            outline = next(
                (item for item in project.outlines if item.chapter_number == chapter.chapter_number),
                None
            )
            chapter_title = outline.title if outline and outline.title else f"第{chapter.chapter_number}章"

            await ingestion_service.ingest_chapter(
                project_id=project_id,
                chapter_number=chapter.chapter_number,
                title=chapter_title,
                content=chapter.selected_version.content,
                summary=chapter.real_summary,
                user_id=current_user.id,
            )
            success_count += 1
            logger.info(
                "章节 %s 嵌入向量重新生成成功",
                chapter.chapter_number,
            )
        except Exception as exc:
            failed_chapters.append(chapter.chapter_number)
            logger.error(
                "章节 %s 嵌入向量重新生成失败: %s",
                chapter.chapter_number,
                exc,
                exc_info=True,
            )

    logger.info(
        "嵌入向量重新生成完成: project=%s 成功=%d 失败=%d",
        project_id,
        success_count,
        len(failed_chapters),
    )

    if failed_chapters:
        logger.warning("以下章节重新生成失败: %s", failed_chapters)

    return await novel_service.get_project_schema(project_id, current_user.id)
