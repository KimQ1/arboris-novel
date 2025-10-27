# Arboris Novel - Docker 部署指南

## 快速启动

### 前置要求

- Docker 20.10+
- Docker Compose 2.0+

### 启动步骤

1. **进入部署目录**
   ```bash
   cd deploy
   ```

2. **检查配置文件**
   
   `.env` 文件已配置好，主要配置项：
   - 端口：`80`
   - 数据库：`SQLite`（数据存储在 Docker 卷 `sqlite-data`）
   - LLM：Gemini 2.5 Pro（已配置 2 个 API Key 轮换）
   - Embedding：Gemini text-embedding-004
   - 章节版本数：`1`（节省配额）

3. **构建并启动容器**
   
   ```bash
   # 方式 1：使用预构建镜像（推荐）
   docker compose up -d
   
   # 方式 2：本地构建镜像
   docker compose -f docker-compose.yml up -d --build
   ```

4. **查看启动日志**
   ```bash
   docker compose logs -f app
   ```

5. **访问应用**
   
   浏览器打开：http://localhost
   
   默认管理员账号：
   - 用户名：`admin`
   - 密码：`ChangeMe123!`

## 数据持久化

### SQLite 模式（默认）

数据存储在 Docker 命名卷 `sqlite-data` 中：

```bash
# 查看卷信息
docker volume inspect deploy_sqlite-data

# 备份数据
docker run --rm -v deploy_sqlite-data:/data -v $(pwd):/backup alpine tar czf /backup/arboris-backup-$(date +%Y%m%d).tar.gz -C /data .

# 恢复数据
docker run --rm -v deploy_sqlite-data:/data -v $(pwd):/backup alpine tar xzf /backup/arboris-backup-YYYYMMDD.tar.gz -C /data
```

### 使用宿主机目录（可选）

修改 `deploy/.env`：
```env
SQLITE_STORAGE_SOURCE=./storage
```

然后重启容器：
```bash
docker compose down
docker compose up -d
```

## 常用命令

### 容器管理

```bash
# 启动
docker compose up -d

# 停止
docker compose down

# 重启
docker compose restart

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f app
```

### 数据库管理

```bash
# 进入容器
docker compose exec app bash

# 查看数据库
cd /app/storage
ls -lh *.db
```

### 更新应用

```bash
# 拉取最新镜像
docker compose pull

# 重启容器
docker compose up -d
```

## 使用 MySQL（可选）

如果需要使用 MySQL 数据库：

1. **修改 `.env` 配置**
   ```env
   DB_PROVIDER=mysql
   MYSQL_PASSWORD=your-strong-password
   MYSQL_ROOT_PASSWORD=your-root-password
   ```

2. **启动 MySQL 服务**
   ```bash
   docker compose --profile mysql up -d
   ```

## 本地构建镜像

如果需要从源码构建：

```bash
# 在项目根目录执行
cd deploy
docker build -f Dockerfile -t arboris-app:local ..

# 修改 docker-compose.yml 中的镜像名
# image: arboris-app:local

# 启动
docker compose up -d
```

## 优化建议

### 减少 API 调用次数

1. **禁用向量嵌入（临时）**
   
   修改 `.env`：
   ```env
   VECTOR_STORE_ENABLED=false
   ```

2. **使用本地 Ollama Embedding**
   
   需要额外部署 Ollama 服务，修改 `.env`：
   ```env
   EMBEDDING_PROVIDER=ollama
   OLLAMA_EMBEDDING_BASE_URL=http://ollama:11434
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
   ```

3. **减少 Chunk 大小**
   
   修改 `.env`：
   ```env
   VECTOR_CHUNK_SIZE=800
   VECTOR_CHUNK_OVERLAP=200
   ```

### 添加更多 API Key

在 `.env` 中用逗号分隔多个 Key：
```env
OPENAI_API_KEY=key1,key2,key3
```

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker compose logs app

# 检查端口占用
netstat -ano | findstr :80

# 重新构建
docker compose down
docker compose up -d --force-recreate
```

### 数据库连接失败

```bash
# 检查数据库文件权限
docker compose exec app ls -lh /app/storage

# 重置数据库（警告：会删除所有数据）
docker compose down -v
docker compose up -d
```

### API 配额超限

- 等待配额重置（每天重置）
- 添加更多 API Key
- 减少版本数：`WRITER_CHAPTER_VERSION_COUNT=1`
- 禁用向量嵌入或使用本地模型

## 安全建议

生产环境部署时：

1. **修改默认密码**
   ```env
   SECRET_KEY=<使用 openssl rand -hex 32 生成>
   ADMIN_DEFAULT_PASSWORD=<强密码>
   ```

2. **禁用调试模式**
   ```env
   DEBUG=false
   ENVIRONMENT=production
   ```

3. **配置 HTTPS**
   
   使用 Nginx 反向代理或 Traefik

4. **限制注册**
   ```env
   ALLOW_USER_REGISTRATION=false
   ```

## 更多信息

- 项目文档：`../docs/`
- 环境变量说明：`.env` 文件注释
- API 文档：http://localhost/api/docs

