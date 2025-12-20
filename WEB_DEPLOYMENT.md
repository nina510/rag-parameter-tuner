# Web 部署指南

## 🎯 部署架构

为了让其他人可以通过网页使用此工具，需要：

1. **前端（静态网页）** - `index.html`，部署到静态文件服务器
2. **后端API服务器** - `app.py`，处理所有 OpenAI API 调用
3. **API密钥安全** - 密钥只存储在服务器端，不暴露给客户端

```
用户浏览器 
  → 前端页面 (index.html)
  → 后端API (app.py) 
  → OpenAI API (使用服务器端的API密钥)
```

## 🔒 API密钥安全处理

### ❌ 不安全的方式（不要这样做）
- 在前端代码中硬编码 API 密钥
- 将 API 密钥传递给前端
- 让用户自己输入 API 密钥（除非是个人使用的工具）

### ✅ 安全的方式（推荐）
- API 密钥只存储在服务器端
- 前端通过 HTTPS 调用后端 API
- 后端验证请求并调用 OpenAI API
- 可以添加速率限制、CORS 保护等

## 📦 部署方案

### 方案1: Render（推荐，免费开始）

#### 后端部署（Render）

1. **创建 `render.yaml` 配置文件**：

```yaml
services:
  - type: web
    name: rag-parameter-tuner-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # 手动在Render Dashboard中设置
      - key: PYTHON_VERSION
        value: 3.11
```

2. **创建 `Procfile`**：

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

3. **更新 `requirements.txt` 添加 gunicorn**：

```
gunicorn>=21.2.0
```

4. **修改 `app.py` 支持 Render**：

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

#### 前端部署（Render Static Site 或其他静态托管）

1. 上传 `index.html` 到 Render Static Site
2. 修改前端 API 地址

### 方案2: Railway

#### 后端部署

1. 连接 GitHub 仓库
2. Railway 自动检测 Python 项目
3. 在环境变量中设置 `OPENAI_API_KEY`
4. 添加 `Procfile`：

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### 方案3: Heroku

1. 创建 `Procfile`
2. 使用 Heroku CLI 部署
3. 设置环境变量：`heroku config:set OPENAI_API_KEY=your-key`

### 方案4: 自有服务器（VPS/云服务器）

#### 使用 Gunicorn + Nginx

1. **安装 Gunicorn**：
```bash
pip install gunicorn
```

2. **启动 Gunicorn**：
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **配置 Nginx 反向代理**：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

4. **使用 systemd 管理服务**（可选）

## 🔧 需要修改的代码

### 1. 修改前端 API 地址

在 `index.html` 中，需要将硬编码的 `localhost:5000` 改为可配置的地址：

```javascript
// 在文件顶部添加
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000'  // 本地开发
    : 'https://your-api-domain.com';  // 生产环境

// 然后在所有 fetch 调用中使用
fetch(`${API_BASE_URL}/api/defaults`)
fetch(`${API_BASE_URL}/api/generate`, ...)
```

或者更好的方式，使用相对路径（如果前后端在同一域名下）：

```javascript
// 使用相对路径，自动适配当前域名
fetch('/api/defaults')
fetch('/api/generate', ...)
```

### 2. 后端 CORS 配置

确保 `app.py` 中的 CORS 配置允许生产域名：

```python
# 开发环境允许所有来源
# 生产环境应该限制为特定域名
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:*",
            "https://your-frontend-domain.com",
            "https://*.render.com",  # 如果使用Render
        ]
    }
})
```

### 3. 添加环境变量支持

```python
# app.py 中
import os

ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:*').split(',')
CORS(app, resources={
    r"/api/*": {"origins": ALLOWED_ORIGINS}
})
```

## 📋 部署检查清单

### 后端部署
- [ ] API 密钥设置为环境变量（不在代码中）
- [ ] 使用生产级 WSGI 服务器（Gunicorn）
- [ ] 配置 CORS 允许前端域名
- [ ] 关闭 debug 模式
- [ ] 设置适当的日志级别
- [ ] 配置 HTTPS（通过反向代理或平台）

### 前端部署
- [ ] 修改 API 地址为生产环境地址
- [ ] 测试所有功能
- [ ] 确保 HTTPS 访问

### 安全措施
- [ ] API 密钥不在代码仓库中
- [ ] 使用 HTTPS 传输
- [ ] 配置速率限制（防止滥用）
- [ ] 考虑添加认证（如果需要限制访问）

## 🚀 快速部署示例（Render）

### 步骤1: 准备文件

1. **创建 `Procfile`**：
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

2. **更新 `requirements.txt`**：
```
gunicorn>=21.2.0
```

3. **修改 `app.py` 最后一行**：
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### 步骤2: 部署后端

1. 在 Render 创建新的 Web Service
2. 连接 GitHub 仓库
3. 设置环境变量：
   - `OPENAI_API_KEY`: 你的 OpenAI API 密钥
4. 部署

### 步骤3: 部署前端

1. 修改 `index.html` 中的 API 地址为后端地址
2. 上传到 Render Static Site 或其他静态托管服务
3. 或者与后端一起部署（Nginx 服务静态文件）

## 💰 成本考虑

- **Render**: 免费层有限制，付费约 $7/月
- **Railway**: 免费层有限制，按使用量付费
- **Heroku**: 不再有免费层
- **自有服务器**: VPS 约 $5-20/月

## ⚠️ 注意事项

1. **API 使用成本**: OpenAI API 是按使用量付费的，需要监控使用情况
2. **速率限制**: 考虑添加速率限制防止滥用
3. **错误处理**: 确保不会暴露敏感信息给前端
4. **日志**: 记录 API 调用以便监控和调试

