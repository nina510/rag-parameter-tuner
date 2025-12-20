# GitHub 部署指南

## 需要上传的文件

### 核心文件（必须）
- `app.py` - Flask 后端 API
- `index.html` - 前端界面
- `requirements.txt` - Python 依赖
- `start.sh` - 启动脚本
- `.gitignore` - Git 忽略文件配置

### 文档文件（推荐）
- `README.md` - 项目说明文档
- `QUICK_START.md` - 快速开始指南
- `DEPLOYMENT.md` - 本部署指南

### 依赖模块
本项目依赖于 `naive_rag.py`，该文件位于 `../naivetest/naive_rag.py`。

**选项1：如果 naive_rag.py 在同一仓库的其他目录**
- 确保仓库中包含 `RAG/naivetest/naive_rag.py`
- `app.py` 中的路径设置会自动处理

**选项2：如果 naive_rag.py 不在同一仓库**
- 需要将 `naive_rag.py` 复制到项目目录，或
- 修改 `app.py` 中的导入路径

## 环境变量配置

### 本地开发
创建 `.env` 文件（已添加到 .gitignore）：
```
OPENAI_API_KEY=your-api-key-here
```

### GitHub Actions / CI/CD
在 GitHub 仓库设置中添加 Secrets：
- `OPENAI_API_KEY`: 你的 OpenAI API 密钥

### 部署到服务器
使用环境变量或 `.env` 文件：
```bash
export OPENAI_API_KEY='your-api-key'
```

## 部署步骤

### 1. 初始化 Git 仓库（如果还没有）
```bash
cd RAG/rag-parameter-tuner
git init
```

### 2. 添加文件
```bash
git add app.py index.html requirements.txt start.sh .gitignore README.md QUICK_START.md DEPLOYMENT.md
```

如果 `naive_rag.py` 在同一仓库：
```bash
git add ../naivetest/naive_rag.py
```

### 3. 提交
```bash
git commit -m "Initial commit: RAG Parameter Tuner"
```

### 4. 添加到远程仓库
```bash
git remote add origin https://github.com/yourusername/your-repo.git
git branch -M main
git push -u origin main
```

## 注意事项

### ✅ 已排除的文件（.gitignore）
- `.env` - 包含敏感信息
- `*.log` - 日志文件
- `__pycache__/` - Python 缓存
- `.DS_Store` - macOS 系统文件

### ⚠️ 需要检查的事项

1. **API 密钥安全**
   - ✅ 代码中已使用环境变量，没有硬编码
   - ✅ `.env` 文件已添加到 .gitignore
   - ⚠️ 确保提交前没有在代码中硬编码 API 密钥

2. **依赖模块路径**
   - 检查 `app.py` 中的 `naivetest_dir` 路径是否正确
   - 如果部署到不同环境，可能需要调整路径

3. **文件大小**
   - FAISS 索引文件可能很大，已注释在 .gitignore 中
   - 如果索引文件需要共享，考虑使用 Git LFS

4. **启动脚本权限**
   ```bash
   chmod +x start.sh
   ```

## 部署后的使用

### 克隆仓库后
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY='your-api-key'

# 启动服务器
python3 app.py
# 或
./start.sh
```

## 常见问题

### Q: naive_rag.py 找不到怎么办？
A: 检查路径设置，确保 `../naivetest/naive_rag.py` 存在，或修改 `app.py` 中的路径。

### Q: 如何在不同端口运行？
A: 修改 `app.py` 最后一行：
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

### Q: 生产环境部署需要注意什么？
A: 
- 关闭 debug 模式
- 使用生产级 WSGI 服务器（如 Gunicorn）
- 配置反向代理（如 Nginx）
- 使用 HTTPS
- 设置适当的安全头

