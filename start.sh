#!/bin/bash
# Render 启动脚本 - 添加日志输出以便诊断问题

echo "=== 启动脚本开始 ==="
echo "当前目录: $(pwd)"
echo "Python 版本: $(python3 --version)"
echo "PORT 环境变量: $PORT"

echo "=== 检查依赖 ==="
python3 -c "import flask; print('Flask:', flask.__version__)" || echo "Flask 导入失败"
python3 -c "import gunicorn; print('Gunicorn 已安装')" || echo "Gunicorn 导入失败"

echo "=== 测试应用导入 ==="
python3 -c "import app; print('应用导入成功')" || {
    echo "应用导入失败！"
    exit 1
}

echo "=== 启动 Gunicorn ==="
exec gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --log-level info --access-logfile - --error-logfile -
