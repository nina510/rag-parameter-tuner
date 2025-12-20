# 纯静态部署方案（GitHub Pages）

如果你确实想要**只用 GitHub Pages**（不依赖其他平台的后端），这里是一个解决方案的说明。

## ⚠️ 重要警告

**这个方案需要用户在浏览器中输入自己的 OpenAI API 密钥**，因为：
1. GitHub Pages 只能托管静态文件，不能运行服务器代码
2. 不能在前端代码中硬编码 API 密钥（会被所有人看到）
3. 唯一的方案是让用户自己提供 API 密钥

## 方案：客户端 API 密钥输入

### 实现步骤

1. **修改前端代码**，添加 API 密钥输入框
2. **直接在浏览器中调用 OpenAI API**
3. **使用 GitHub Actions 构建检索索引**（可选）

### 代码修改示例

在 `index.html` 中添加：

```html
<!-- 在页面顶部添加 API 密钥输入 -->
<div id="api-key-prompt" style="display: none; padding: 20px; background: #fff3cd; border: 1px solid #ffc107;">
    <h3>请输入您的 OpenAI API 密钥</h3>
    <input type="password" id="user-api-key" placeholder="sk-..." style="width: 300px; padding: 8px;">
    <button onclick="saveApiKey()">保存</button>
    <p style="font-size: 12px; color: #666;">
        密钥仅存储在浏览器本地，不会上传到服务器。
        <a href="https://platform.openai.com/api-keys" target="_blank">获取 API 密钥</a>
    </p>
</div>

<script>
    // 从 localStorage 读取 API 密钥
    function getApiKey() {
        return localStorage.getItem('openai_api_key') || '';
    }
    
    function saveApiKey() {
        const key = document.getElementById('user-api-key').value.trim();
        if (key) {
            localStorage.setItem('openai_api_key', key);
            document.getElementById('api-key-prompt').style.display = 'none';
            location.reload();
        }
    }
    
    // 检查是否有 API 密钥
    if (!getApiKey()) {
        document.getElementById('api-key-prompt').style.display = 'block';
    }
</script>
```

然后修改 API 调用，直接在浏览器中调用 OpenAI：

```javascript
// 替换原来的 fetch('/api/generate', ...)
async function generateAnswerDirect() {
    const apiKey = getApiKey();
    if (!apiKey) {
        alert('请先输入 OpenAI API 密钥');
        return;
    }
    
    // 直接调用 OpenAI API（需要先在浏览器中实现 RAG 逻辑）
    // 这需要将整个 RAG 逻辑移植到前端...
}
```

## ❌ 这个方案的问题

1. **需要完整的前端 RAG 实现**
   - 需要在浏览器中实现文档检索、向量搜索等
   - 需要在浏览器中加载和搜索 FAISS 索引
   - 代码量巨大，性能差

2. **用户体验差**
   - 用户必须有 OpenAI 账号
   - 需要输入 API 密钥
   - 无法提供"开箱即用"的服务

3. **技术限制**
   - FAISS 索引文件很大（几MB到几十MB）
   - 需要下载到浏览器，加载慢
   - 浏览器内存限制

4. **功能受限**
   - 某些检索器（如 BM25、MedCPT）需要服务器端处理
   - 无法实现复杂的检索逻辑

## ✅ 推荐方案

**前端：GitHub Pages + 后端：Render（免费层）**

这是最佳平衡：
- 前端免费（GitHub Pages）
- 后端免费（Render 免费层）
- 用户体验好（无需输入 API 密钥）
- 功能完整
- 安全可靠

## 📝 结论

虽然理论上可以只用 GitHub Pages，但：
- 需要大幅重构代码
- 用户体验差
- 功能受限
- 性能问题

**推荐使用混合方案**：GitHub Pages（前端）+ Render（后端），这样既接近"只用 GitHub"的愿望，又能保证功能和体验。

