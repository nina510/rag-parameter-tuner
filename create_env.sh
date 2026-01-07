#!/bin/bash
# 创建 .env 文件的辅助脚本

echo "创建 .env 文件..."
echo ""
read -p "请输入您的 OpenAI API 密钥（sk-proj-开头）: " api_key

if [ -z "$api_key" ]; then
    echo "错误：API 密钥不能为空"
    exit 1
fi

echo "OPENAI_API_KEY=$api_key" > .env
echo ""
echo "✓ .env 文件已创建成功！"
echo ""
echo "验证文件内容："
cat .env

