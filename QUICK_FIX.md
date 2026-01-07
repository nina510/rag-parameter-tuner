# 快速修复：浏览器显示 JavaScript 代码

## 问题症状

浏览器显示大段 JavaScript 代码而不是正常的网页界面。

## 立即解决方案

### 步骤 1: 停止当前服务器

如果服务器正在运行，按 `Ctrl+C` 停止。

### 步骤 2: 清除浏览器缓存

**重要**：必须清除浏览器缓存！

- **Chrome/Edge**: 
  - 按 `Ctrl+Shift+Delete`
  - 选择"缓存的图片和文件"
  - 点击"清除数据"
  
- **Firefox**:
  - 按 `Ctrl+Shift+Delete`
  - 选择"缓存"
  - 点击"立即清除"

或者使用**硬刷新**：
- **Windows/Linux**: `Ctrl+F5` 或 `Ctrl+Shift+R`
- **Mac**: `Cmd+Shift+R`

### 步骤 3: 重新启动服务器

```bash
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
python3 app.py
```

### 步骤 4: 访问应用

**重要**：确保使用正确的 URL

✅ **正确**: `http://localhost:5000`
❌ **错误**: `file:///...` 或直接打开 HTML 文件

在浏览器地址栏输入：`http://localhost:5000`

### 步骤 5: 如果还是不行

1. **检查浏览器控制台**（按 F12）：
   - 查看 Console 标签是否有错误
   - 查看 Network 标签，检查 `index.html` 的响应
   - 确认 `Content-Type: text/html`

2. **尝试无痕/隐私模式**：
   - Chrome: `Ctrl+Shift+N`
   - Firefox: `Ctrl+Shift+P`
   - 然后访问 `http://localhost:5000`

3. **检查服务器日志**：
   - 确认服务器正在运行
   - 查看是否有错误信息

## 验证修复

修复后，你应该看到：
- ✅ 正常的网页界面（不是代码）
- ✅ 左侧有参数控制面板
- ✅ 右侧有答案显示区域
- ✅ 浏览器地址栏显示 `http://localhost:5000`

## 如果问题仍然存在

请提供以下信息：
1. 浏览器地址栏显示的完整 URL
2. 浏览器控制台（F12）的错误信息
3. 服务器终端的输出

