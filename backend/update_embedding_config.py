"""
临时脚本：更新嵌入模型配置
使用方法：python update_embedding_config.py
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select, update
from app.db.session import AsyncSessionLocal
from app.models.system_config import SystemConfig


async def update_embedding_config():
    """更新嵌入模型配置为 Ollama 本地模型（免费方案）"""
    async with AsyncSessionLocal() as session:
        # 方案选择
        print("请选择嵌入模型配置方案：")
        print("1. 使用 Ollama 本地模型（免费，需要先安装 Ollama）")
        print("2. 使用 OpenAI 兼容 API（需要 API Key）")
        print("3. 禁用嵌入功能（不推荐，会影响 RAG 检索）")
        
        choice = input("请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            # Ollama 方案
            configs = [
                ("embedding.provider", "ollama"),
                ("ollama.embedding_base_url", "http://localhost:11434"),
                ("ollama.embedding_model", "nomic-embed-text:latest"),
            ]
            print("\n配置为 Ollama 本地模型...")
            print("注意：请确保已安装 Ollama 并运行 'ollama pull nomic-embed-text'")
            
        elif choice == "2":
            # OpenAI 兼容方案
            api_key = input("请输入 API Key: ").strip()
            base_url = input("请输入 Base URL (留空使用 OpenAI 官方): ").strip()
            model = input("请输入模型名称 (默认 text-embedding-3-small): ").strip() or "text-embedding-3-small"
            
            configs = [
                ("embedding.provider", "openai"),
                ("embedding.api_key", api_key),
                ("embedding.base_url", base_url or None),
                ("embedding.model", model),
            ]
            print("\n配置为 OpenAI 兼容 API...")
            
        elif choice == "3":
            print("\n警告：禁用嵌入功能会影响章节上下文检索功能")
            confirm = input("确认禁用？(yes/no): ").strip().lower()
            if confirm != "yes":
                print("已取消")
                return
            # 这里可以设置一个标志，但实际上系统会在 API Key 无效时自动跳过
            print("建议：保持配置不变，系统会在嵌入失败时自动降级")
            return
        else:
            print("无效选项")
            return
        
        # 更新配置
        for key, value in configs:
            if value is None:
                continue
                
            # 检查配置是否存在
            result = await session.execute(
                select(SystemConfig).where(SystemConfig.key == key)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新现有配置
                await session.execute(
                    update(SystemConfig)
                    .where(SystemConfig.key == key)
                    .values(value=value)
                )
                print(f"✓ 更新配置: {key} = {value}")
            else:
                # 创建新配置
                new_config = SystemConfig(key=key, value=value)
                session.add(new_config)
                print(f"✓ 创建配置: {key} = {value}")
        
        await session.commit()
        print("\n配置更新成功！请重启后端服务以使配置生效。")


if __name__ == "__main__":
    asyncio.run(update_embedding_config())

