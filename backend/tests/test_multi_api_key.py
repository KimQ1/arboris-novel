"""测试多 API Key 轮询功能"""
import pytest
from app.services.llm_service import LLMService


class TestMultiAPIKey:
    """测试多 API Key 解析和轮询功能"""

    def test_parse_single_key(self):
        """测试解析单个 API Key"""
        service = LLMService(session=None)
        
        # 单个 Key
        keys = service._parse_api_keys("AIzaSyABC123")
        assert len(keys) == 1
        assert keys[0] == "AIzaSyABC123"

    def test_parse_multiple_keys_newline(self):
        """测试解析多个 API Key（换行符分隔）"""
        service = LLMService(session=None)
        
        # 换行符分隔
        keys = service._parse_api_keys("AIzaSyABC123\nAIzaSyDEF456\nAIzaSyGHI789")
        assert len(keys) == 3
        assert keys[0] == "AIzaSyABC123"
        assert keys[1] == "AIzaSyDEF456"
        assert keys[2] == "AIzaSyGHI789"

    def test_parse_multiple_keys_comma(self):
        """测试解析多个 API Key（逗号分隔）"""
        service = LLMService(session=None)
        
        # 逗号分隔
        keys = service._parse_api_keys("AIzaSyABC123, AIzaSyDEF456, AIzaSyGHI789")
        assert len(keys) == 3
        assert keys[0] == "AIzaSyABC123"
        assert keys[1] == "AIzaSyDEF456"
        assert keys[2] == "AIzaSyGHI789"

    def test_parse_multiple_keys_mixed(self):
        """测试解析多个 API Key（混合分隔符）"""
        service = LLMService(session=None)
        
        # 混合分隔符
        keys = service._parse_api_keys("AIzaSyABC123, AIzaSyDEF456\nAIzaSyGHI789")
        assert len(keys) == 3
        assert keys[0] == "AIzaSyABC123"
        assert keys[1] == "AIzaSyDEF456"
        assert keys[2] == "AIzaSyGHI789"

    def test_parse_keys_with_whitespace(self):
        """测试解析带空格的 API Key"""
        service = LLMService(session=None)
        
        # 带空格
        keys = service._parse_api_keys("  AIzaSyABC123  \n  AIzaSyDEF456  ")
        assert len(keys) == 2
        assert keys[0] == "AIzaSyABC123"
        assert keys[1] == "AIzaSyDEF456"

    def test_parse_empty_key(self):
        """测试解析空 API Key"""
        service = LLMService(session=None)
        
        # 空字符串
        keys = service._parse_api_keys("")
        assert len(keys) == 0
        
        # None
        keys = service._parse_api_keys(None)
        assert len(keys) == 0

    def test_parse_keys_with_empty_lines(self):
        """测试解析包含空行的 API Key"""
        service = LLMService(session=None)
        
        # 包含空行
        keys = service._parse_api_keys("AIzaSyABC123\n\nAIzaSyDEF456\n")
        assert len(keys) == 2
        assert keys[0] == "AIzaSyABC123"
        assert keys[1] == "AIzaSyDEF456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

