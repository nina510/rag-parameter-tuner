# 故障排除指南

## 问题：浏览器显示大段 JavaScript 代码而不是正常页面

### 原因

这通常是因为：
1. **直接打开了 HTML 文件**（使用 `file://` 协议）而不是通过服务器访问
2. **Content-Type 头未正确设置**（已修复）
3. **JavaScript 代码在 `<script>` 标签外**

### 解决方案

#### ✅ 方案 1: 通过 Flask 服务器访问（推荐）

**不要直接双击打开 `index.html` 文件！**

正确的方式：

1. **启动 Flask 服务器**：
   ```bash
   cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
   python3 app.py
   ```

2. **在浏览器中访问**：
   ```
   http://localhost:5000
   ```

   不要使用 `file:///` 路径！

#### ✅ 方案 2: 使用 Python HTTP 服务器（临时测试）

如果 Flask 服务器有问题，可以使用简单的 HTTP 服务器：

```bash
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
python3 -m http.server 8000
```

然后访问：`http://localhost:8000`

### 如何判断问题

#### 检查浏览器地址栏

- ❌ **错误**：`file:///home/yw28498/.../index.html`
- ✅ **正确**：`http://localhost:5000` 或 `http://127.0.0.1:5000`

#### 检查服务器是否运行

```bash
# 检查端口 5000 是否被占用
lsof -i :5000

# 或
netstat -an | grep 5000
```

### 常见错误

#### 错误 1: 直接打开 HTML 文件

**症状**：浏览器地址栏显示 `file:///...`

**解决**：必须通过 HTTP 服务器访问

#### 错误 2: 服务器未启动

**症状**：浏览器显示 "无法连接" 或 "ERR_CONNECTION_REFUSED"

**解决**：
```bash
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
python3 app.py
```

#### 错误 3: 端口被占用

**症状**：启动时显示 "Address already in use"

**解决**：
```bash
# 查找占用端口的进程
lsof -i :5000

# 终止进程
kill <PID>

# 或使用其他端口
export PORT=8080
python3 app.py
```

### 验证修复

修复后，你应该看到：
- ✅ 正常的网页界面（不是代码）
- ✅ 左侧有参数控制面板
- ✅ 右侧有答案显示区域
- ✅ 浏览器地址栏显示 `http://localhost:5000`

### 如果问题仍然存在

1. **清除浏览器缓存**：
   - Chrome/Edge: `Ctrl+Shift+Delete` → 清除缓存
   - Firefox: `Ctrl+Shift+Delete` → 清除缓存

2. **检查浏览器控制台**：
   - 按 `F12` 打开开发者工具
   - 查看 Console 标签是否有错误

3. **检查网络请求**：
   - 在开发者工具的 Network 标签
   - 查看 `index.html` 的响应头
   - 确认 `Content-Type: text/html`

4. **重启服务器**：
   ```bash
   # 停止当前服务器（Ctrl+C）
   # 重新启动
   python3 app.py
   ```

