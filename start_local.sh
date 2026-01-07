#!/bin/bash
# 本地启动脚本

cd "$(dirname "$0")"

echo "=========================================="
echo "RAG Parameter Tuner - 本地启动"
echo "=========================================="

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: OPENAI_API_KEY 未设置"
    echo "请设置: export OPENAI_API_KEY='your-key'"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "❌ Flask 未安装，正在安装依赖..."
    pip install -r requirements.txt
fi

# 设置默认端口
export PORT=${PORT:-5000}
export FLASK_DEBUG=${FLASK_DEBUG:-True}

echo ""
echo "启动应用..."
echo "访问地址: http://localhost:$PORT"
echo "按 Ctrl+C 停止"
echo "=========================================="
echo ""

# 启动应用
python3 app.py
