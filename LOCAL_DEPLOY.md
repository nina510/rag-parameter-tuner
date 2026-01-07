# 本地部署指南

## 快速开始

### 1. 安装依赖

```bash
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
pip install -r requirements.txt
```

### 2. 设置环境变量

创建 `.env` 文件（如果还没有）：

```bash
# 在项目根目录创建 .env 文件
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
FLASK_DEBUG=True
PORT=5000
EOF
```

或者直接设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export FLASK_DEBUG="True"
export PORT=5000
```

### 3. 启动应用

#### 方式 A: 直接运行（开发模式）

```bash
python3 app.py
```

应用会在 `http://localhost:5000` 启动。

#### 方式 B: 使用 Gunicorn（生产模式）

```bash
gunicorn app:app --bind 0.0.0.0:5000 --timeout 120 --workers 1 --log-level info
```

### 4. 访问应用

打开浏览器访问：`http://localhost:5000`

## 详细步骤

### 步骤 1: 检查 Python 版本

```bash
python3 --version
# 需要 Python 3.9 或更高版本
```

### 步骤 2: 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤 3: 检查必要文件

确保以下文件存在：
- ✅ `app.py` - Flask 应用
- ✅ `index.html` - 前端页面
- ✅ `naive_rag.py` - RAG 逻辑
- ✅ `load.py` - 文档加载
- ✅ `corpus/` - 文档目录
  - `corpus/csv/LongCovid.csv`
  - `corpus/txt/LongCovid/*.txt` (8个文件)

### 步骤 4: 设置 API 密钥

**重要**: 必须设置 `OPENAI_API_KEY` 环境变量。

```bash
# 方法 1: 使用 .env 文件
echo "OPENAI_API_KEY=sk-..." > .env

# 方法 2: 直接设置环境变量
export OPENAI_API_KEY="sk-..."
```

### 步骤 5: 启动应用

#### 开发模式（带自动重载）

```bash
python3 app.py
```

#### 生产模式（使用 Gunicorn）

```bash
gunicorn app:app --bind 0.0.0.0:5000 --timeout 120 --workers 1 --log-level info --reload
```

### 步骤 6: 验证部署

1. 打开浏览器访问 `http://localhost:5000`
2. 应该能看到前端界面
3. 测试生成答案功能

## 常见问题

### 问题 1: 端口被占用

如果 5000 端口被占用，可以修改端口：

```bash
# 方法 1: 修改环境变量
export PORT=8080
python3 app.py

# 方法 2: 直接指定端口
python3 app.py  # 默认 5000
# 或
gunicorn app:app --bind 0.0.0.0:8080
```

### 问题 2: 依赖安装失败

如果某些依赖安装失败：

```bash
# 尝试升级 pip
pip install --upgrade pip

# 单独安装失败的包
pip install package-name

# 如果 torch 安装失败，可以只安装 CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 问题 3: 导入错误

如果遇到 `ModuleNotFoundError`：

```bash
# 确保在项目根目录
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner

# 检查 Python 路径
python3 -c "import sys; print(sys.path)"

# 测试导入
python3 -c "import app; print('OK')"
```

### 问题 4: corpus 文件缺失

如果提示找不到 corpus 文件：

```bash
# 检查文件是否存在
ls -la corpus/csv/LongCovid.csv
ls -la corpus/txt/LongCovid/*.txt

# 如果文件不存在，需要从原始位置复制
# （根据你的项目结构调整路径）
```

## 启动脚本

可以创建一个启动脚本 `start_local.sh`：

```bash
#!/bin/bash
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: OPENAI_API_KEY 未设置"
    echo "请设置: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# 启动应用
echo "启动 Flask 应用..."
python3 app.py
```

使用：

```bash
chmod +x start_local.sh
./start_local.sh
```

## 测试

### 测试 API 端点

```bash
# 健康检查
curl http://localhost:5000/api/health

# 获取默认参数
curl http://localhost:5000/api/defaults

# 生成答案（需要 POST）
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of Long COVID?",
    "chunk_size": 1200,
    "chunk_overlap": 600,
    "model": "gpt-4o"
  }'
```

## 日志

应用日志会输出到控制台。如果需要保存日志：

```bash
# 保存到文件
python3 app.py 2>&1 | tee app.log

# 或使用 Gunicorn
gunicorn app:app --bind 0.0.0.0:5000 --log-file app.log
```

## 停止应用

- 如果使用 `python3 app.py`：按 `Ctrl+C`
- 如果使用 Gunicorn：找到进程并终止

```bash
# 查找进程
ps aux | grep gunicorn

# 终止进程
kill <PID>
```

## 下一步

部署成功后，你可以：
1. 在浏览器中访问 `http://localhost:5000`
2. 测试不同的参数组合
3. 查看生成的答案和引用

