#!/bin/bash

# RAG 参数调节工具启动脚本

echo "=========================================="
echo "RAG 参数调节工具"
echo "=========================================="

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3"
    exit 1
fi

# 检查并加载环境变量
if [ -f .env ]; then
    echo "从 .env 文件加载环境变量..."
    export $(grep -v '^#' .env | xargs)
fi

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: OPENAI_API_KEY 环境变量未设置"
    echo "请设置: export OPENAI_API_KEY='your-api-key'"
    echo "或者创建 .env 文件并添加: OPENAI_API_KEY=your-api-key"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ OPENAI_API_KEY 已设置"
fi

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "安装依赖..."
    pip3 install -r requirements.txt
fi

# 启动服务器
echo ""
echo "启动 Flask 服务器..."
echo "服务器地址: http://localhost:5000"
echo "前端界面: 在浏览器中打开 index.html"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "=========================================="

python3 app.py

