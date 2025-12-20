# 快速部署指南

## 🚀 最简单的部署方式（Render - 免费开始）

### 步骤1: 准备代码

所有必要的文件已经准备好了：
- ✅ `Procfile` - Render 启动配置
- ✅ `requirements.txt` - 包含所有依赖
- ✅ `app.py` - 已支持生产环境配置
- ✅ `index.html` - API 地址已配置为自动检测

### 步骤2: 推送到 GitHub

```bash
git add .
git commit -m "Prepare for web deployment"
git push origin main
```

### 步骤3: 在 Render 部署后端

1. 访问 [render.com](https://render.com) 并登录
2. 点击 "New +" → "Web Service"
3. 连接你的 GitHub 仓库
4. 配置：
   - **Name**: `rag-parameter-tuner-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
   - **Instance Type**: Free（或付费）

5. **添加环境变量**：
   - `OPENAI_API_KEY`: 你的 OpenAI API 密钥
   - `FLASK_DEBUG`: `False`
   - `ALLOWED_ORIGINS`: 你的前端域名（如果前后端分离）

6. 点击 "Create Web Service"

7. 等待部署完成，记下后端 URL（如：`https://rag-parameter-tuner-api.onrender.com`）

### 步骤4: 部署前端

#### 选项A: Render Static Site（推荐）

1. 在 Render 创建 "Static Site"
2. 连接 GitHub 仓库
3. 配置：
   - **Build Command**: 留空（静态文件不需要构建）
   - **Publish Directory**: `RAG/rag-parameter-tuner`（或上传 index.html 的位置）

4. 在 `index.html` 开头添加配置（如果前后端在不同域名）：
```html
<script>
    // 配置后端 API 地址
    window.API_BASE_URL = 'https://rag-parameter-tuner-api.onrender.com/api';
</script>
```

#### 选项B: 与后端一起部署

修改 `app.py` 添加静态文件服务：
```python
@app.route('/')
def index():
    return app.send_static_file('index.html')

# 然后在启动时指定 static_folder
app = Flask(__name__, static_folder='.', static_url_path='')
```

### 步骤5: 测试

1. 访问前端 URL
2. 测试所有功能
3. 检查浏览器控制台是否有错误

## 🔒 API 密钥安全

### ✅ 正确做法（已配置）

- ✅ API 密钥存储在服务器环境变量中
- ✅ `.env` 文件已在 `.gitignore` 中
- ✅ 代码中通过 `os.environ.get('OPENAI_API_KEY')` 读取
- ✅ 前端永远不接触 API 密钥

### ❌ 错误做法

- ❌ 在前端代码中硬编码 API 密钥
- ❌ 将 API 密钥提交到 Git 仓库
- ❌ 在 URL 参数中传递 API 密钥

## 💰 成本估算

### Render 免费层限制
- 15 分钟无活动后休眠
- 启动需要几秒钟
- 每月 750 小时运行时间

### OpenAI API 成本
- 按实际使用量计费
- GPT-4: ~$0.03/1K tokens（输入），~$0.06/1K tokens（输出）
- 建议设置使用量限制和监控

## 🛠️ 其他部署选项

### Railway
```bash
# 1. 安装 Railway CLI
npm i -g @railway/cli

# 2. 登录
railway login

# 3. 初始化项目
railway init

# 4. 设置环境变量
railway variables set OPENAI_API_KEY=your-key

# 5. 部署
railway up
```

### 自有服务器（VPS）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量
export OPENAI_API_KEY='your-key'

# 3. 使用 Gunicorn 启动
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 4. 配置 Nginx 反向代理（推荐）
# 参见 WEB_DEPLOYMENT.md
```

## 📝 部署后检查清单

- [ ] 后端 API 可以访问（测试 `/api/health`）
- [ ] 前端页面可以加载
- [ ] 前端可以成功调用后端 API
- [ ] API 密钥已正确设置
- [ ] HTTPS 已启用（生产环境必需）
- [ ] CORS 配置正确（如果前后端分离）
- [ ] 错误处理正常工作
- [ ] 日志记录正常

## 🔍 故障排除

### 问题：CORS 错误
**解决方案**：检查 `ALLOWED_ORIGINS` 环境变量，确保包含前端域名

### 问题：API 调用失败
**解决方案**：
1. 检查后端 URL 是否正确
2. 检查浏览器控制台的错误信息
3. 检查后端日志

### 问题：API 密钥错误
**解决方案**：
1. 确认环境变量已正确设置
2. 重启服务使环境变量生效
3. 检查变量名拼写（`OPENAI_API_KEY`）

### 问题：服务启动失败
**解决方案**：
1. 检查 `requirements.txt` 是否完整
2. 查看部署日志
3. 确认 Python 版本兼容（建议 3.9+）

