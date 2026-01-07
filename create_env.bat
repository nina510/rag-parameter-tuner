@echo off
REM 创建 .env 文件的辅助脚本

echo 创建 .env 文件...
echo.
set /p api_key="请输入您的 OpenAI API 密钥（sk-proj-开头）: "

if "%api_key%"=="" (
    echo 错误：API 密钥不能为空
    pause
    exit /b 1
)

echo OPENAI_API_KEY=%api_key% > .env
echo.
echo .env 文件已创建成功！
echo.
echo 验证文件内容：
type .env
echo.
pause

