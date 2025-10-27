# 多 API Key 轮询配置指南

## 功能说明

系统支持配置多个 LLM API Key，当一个 Key 达到配额限制（429 错误）时，会自动切换到下一个 Key 继续请求。

## 配置方法

### 方法 1：通过后台管理界面配置

1. 登录后台管理界面
2. 进入 **系统配置** 页面
3. 找到 `llm.api_key` 配置项
4. 在值中输入多个 API Key，每行一个或用逗号分隔

**示例**（换行符分隔）：
```
AIzaSyABC123...
AIzaSyDEF456...
AIzaSyGHI789...
```

**示例**（逗号分隔）：
```
AIzaSyABC123..., AIzaSyDEF456..., AIzaSyGHI789...
```

### 方法 2：通过环境变量配置

在 `.env` 文件中设置：

```bash
# 多个 Key 用换行符分隔
OPENAI_API_KEY="AIzaSyABC123...
AIzaSyDEF456...
AIzaSyGHI789..."
```

或者用逗号分隔：

```bash
OPENAI_API_KEY="AIzaSyABC123..., AIzaSyDEF456..., AIzaSyGHI789..."
```

### 方法 3：通过数据库直接配置

```sql
UPDATE system_configs 
SET value = 'AIzaSyABC123...
AIzaSyDEF456...
AIzaSyGHI789...'
WHERE key = 'llm.api_key';
```

## 工作原理

1. **轮询策略**：系统使用 Round-Robin（轮询）策略选择 API Key
2. **自动切换**：当遇到 429 错误时，自动切换到下一个 Key
3. **失败处理**：如果所有 Key 都达到限制，返回错误提示

## 日志示例

成功切换时的日志：

```
INFO: 尝试使用 API Key #1/3 (索引: 0)
WARNING: API Key #0 达到配额限制 (429)，尝试下一个 Key
INFO: 尝试使用 API Key #2/3 (索引: 1)
INFO: API Key #1 调用成功，下次将使用 Key #2
```

所有 Key 都失败时的日志：

```
ERROR: 所有 3 个 API Key 都已达到配额限制
```

## 最佳实践

1. **建议配置 3-5 个 Key**：平衡成本和可用性
2. **使用不同的 Google 账号**：每个账号都有独立的免费配额
3. **监控使用情况**：定期检查日志，了解 Key 的使用情况
4. **混合使用**：可以配置免费 Key + 付费 Key，优先使用免费配额

## 注意事项

- 只有系统级配置支持多 Key，用户自定义配置暂不支持
- 所有 Key 必须使用相同的 `base_url` 和 `model`
- Key 的顺序会影响使用优先级，第一个 Key 会被优先使用

## 故障排查

### 问题：配置了多个 Key 但没有轮询

**检查**：
1. 确认 Key 之间用换行符或逗号分隔
2. 检查日志中是否显示 "尝试使用 API Key #X/Y"
3. 确认没有其他语法错误

### 问题：所有 Key 都很快达到限制

**解决方案**：
1. 增加更多 API Key
2. 考虑升级到付费计划
3. 使用本地 Ollama 作为备选方案

## 示例配置

### Gemini 免费版（每天 50 次/Key）

```
AIzaSyABC123...  # 账号1
AIzaSyDEF456...  # 账号2
AIzaSyGHI789...  # 账号3
```

总配额：150 次/天

### 混合配置（免费 + 付费）

```
AIzaSyABC123...  # 免费账号1
AIzaSyDEF456...  # 免费账号2
sk-proj-abc123...  # OpenAI 付费账号
```

优先使用免费配额，用完后自动切换到付费账号。

