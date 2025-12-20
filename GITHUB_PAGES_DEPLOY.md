# GitHub Pages 部署说明

## ⚠️ GitHub Pages 的局限性

**GitHub Pages 只能部署静态网站**（HTML、CSS、JavaScript），**不能运行服务器端代码**（如 Python Flask 应用）。

### 为什么不能直接用 GitHub Pages？

1. **后端代码需要运行环境**
   - `app.py` 是 Flask 应用，需要在服务器上运行 Python
   - 需要处理 API 请求、调用 OpenAI API
   - GitHub Pages 只提供静态文件托管，不提供 Python 运行环境

2. **API 密钥安全**
   - 即使能用 GitHub Pages，也不能在前端代码中暴露 API 密钥
   - 后端服务器可以安全地存储 API 密钥，前端通过 API 调用

### 当前的架构

```
用户浏览器 
  → 前端页面 (index.html) ← 可以用 GitHub Pages
  → 后端API (app.py) ← 必须用其他平台（Render/Railway/Heroku等）
  → OpenAI API
```

## 🎯 混合部署方案（推荐）

### 方案：前端用 GitHub Pages + 后端用免费平台

这是最接近"只用 GitHub"的方案：

#### 步骤1：前端部署到 GitHub Pages

1. 在 GitHub 仓库设置中：
   - 进入 Settings → Pages
   - Source: 选择 `main` 分支，`/` 或 `/root` 目录
   - 保存

2. 前端访问地址：
   - `https://yourusername.github.io/rag-parameter-tuner/`

#### 步骤2：修改前端 API 地址

在 `index.html` 中配置后端 API 地址（部署后端后获取）：

```html
<script>
    // 配置后端 API 地址（部署后端后替换为实际地址）
    window.API_BASE_URL = 'https://your-backend.onrender.com/api';
</script>
```

#### 步骤3：后端部署（仍然需要其他平台）

后端仍然需要部署到能运行 Python 的平台，因为：
- GitHub Pages 无法运行 Flask
- 需要环境变量存储 API 密钥
- 需要持续运行的服务器进程

**推荐平台**（都有免费层）：
- **Render**（最简单，免费层有15分钟休眠限制）
- **Railway**（按使用量计费）
- **Fly.io**（有免费额度）

## 📋 完整部署步骤

### 1. 准备前端（GitHub Pages）

```bash
# 在仓库根目录，确保 index.html 在正确位置
# GitHub Pages 会自动提供静态文件服务
```

**前端配置**（在 `index.html` 开头添加）：
```html
<script>
    // 自动检测环境
    const API_BASE = (function() {
        const hostname = window.location.hostname;
        
        // GitHub Pages 环境
        if (hostname.includes('github.io')) {
            // 使用部署的后端地址
            return 'https://your-backend.onrender.com/api';
        }
        
        // 本地开发
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:5000/api';
        }
        
        // 默认（同域部署）
        return '/api';
    })();
</script>
```

### 2. 部署后端（Render - 免费开始）

1. 访问 [render.com](https://render.com)，用 GitHub 账号登录
2. 创建新的 Web Service
3. 连接 GitHub 仓库
4. 配置：
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. 添加环境变量：`OPENAI_API_KEY`
6. 部署完成后获取后端 URL（如：`https://rag-tuner.onrender.com`）

### 3. 更新前端 API 地址

在前端代码中设置后端 URL：
```html
<script>
    window.API_BASE_URL = 'https://rag-tuner.onrender.com/api';
</script>
```

### 4. 配置 CORS

确保后端允许 GitHub Pages 域名访问：

在 `app.py` 中或环境变量中设置：
```python
ALLOWED_ORIGINS=https://yourusername.github.io,https://yourusername.github.io/rag-parameter-tuner
```

## 🔄 纯静态方案（不推荐）

如果想**完全避免后端服务器**，有几个方案，但都有问题：

### ❌ 方案1：前端直接调用 OpenAI API
- **问题**：必须在 JavaScript 中硬编码 API 密钥
- **风险**：API 密钥会暴露给所有用户，非常不安全
- **后果**：任何人都能看到并使用你的 API 密钥，导致费用激增

### ❌ 方案2：客户端输入 API 密钥
- **问题**：需要用户自己提供 API 密钥
- **缺点**：用户体验差，需要用户有 OpenAI 账号
- **不适用**：无法提供"开箱即用"的服务

### ✅ 推荐的混合方案
- 前端：GitHub Pages（免费，简单）
- 后端：Render/Railway（有免费层，安全）
- **优点**：接近"只用 GitHub"，同时保证安全和功能完整

## 💡 为什么推荐这个方案？

1. **前端免费**：GitHub Pages 完全免费
2. **后端简单**：Render 免费层足够使用（有休眠限制）
3. **安全**：API 密钥只在服务器端，不会暴露
4. **易维护**：代码都在 GitHub，前端自动部署

## 📝 总结

- **GitHub Pages**：可以部署前端 ✅
- **后端服务器**：必须用其他平台 ❌（GitHub Pages 无法运行 Python）
- **最佳方案**：GitHub Pages（前端）+ Render（后端，免费层）

如果你想尽可能"只用 GitHub"，前端可以部署到 GitHub Pages，但后端仍然需要其他平台，因为这是技术限制，不是选择问题。

