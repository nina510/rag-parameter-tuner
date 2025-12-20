# Render 部署故障排除

## 问题：No open ports detected

如果部署时看到 "No open ports detected"，可能的原因和解决方案：

### 原因 1: Gunicorn 配置问题

**检查 Procfile**：
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --log-level info
```

确保：
- ✅ 使用 `$PORT` 环境变量（Render 会自动设置）
- ✅ 绑定到 `0.0.0.0`（不是 `127.0.0.1`）
- ✅ workers 数量不要太多（免费层建议 1）

### 原因 2: 应用启动失败

**检查日志**：
1. 在 Render 控制台点击 "Logs" 标签
2. 查看是否有 Python 错误
3. 检查是否有导入错误或依赖缺失

**常见错误**：
- `ModuleNotFoundError`: 检查 `requirements.txt` 是否包含所有依赖
- `ImportError`: 检查代码中的导入路径
- `AttributeError`: 检查代码逻辑

### 原因 3: 端口绑定延迟

有时应用需要更长时间启动，特别是：
- 首次部署（需要下载依赖）
- 加载大型文件（如 FAISS 索引）
- 初始化模型（如 sentence-transformers）

**解决方案**：
- 等待 2-3 分钟
- 检查日志确认应用是否正在启动
- 如果一直失败，检查启动命令

### 原因 4: 环境变量未设置

**检查必需的环境变量**：
- `OPENAI_API_KEY`: 必须设置
- `PORT`: Render 自动设置，不需要手动设置

### 调试步骤

1. **查看完整日志**
   ```
   在 Render 控制台 → Logs → 查看所有输出
   ```

2. **测试本地启动**
   ```bash
   cd /path/to/rag-parameter-tuner
   export PORT=5000
   gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
   ```

3. **检查应用代码**
   - 确保 `app.py` 中有正确的 Flask 应用实例
   - 确保没有在 `if __name__ == '__main__'` 中硬编码端口

4. **简化启动命令**
   如果复杂命令失败，尝试简化：
   ```
   web: gunicorn app:app
   ```
   Render 会自动处理端口绑定

### 当前配置检查

✅ **Procfile**: `web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --log-level info`

✅ **app.py**: 已配置为从环境变量读取端口

✅ **requirements.txt**: 包含 `gunicorn`

### 如果问题持续

1. **检查 Render 文档**: https://render.com/docs/web-services#port-binding
2. **查看应用日志**: 在 Render 控制台的 Logs 标签
3. **尝试手动指定端口**（不推荐，但可以测试）:
   ```
   web: gunicorn app:app --bind 0.0.0.0:10000 --timeout 120
   ```
   注意：Render 会自动设置 `$PORT`，手动指定可能导致冲突

4. **联系 Render 支持**: 如果以上都不行，可能是平台问题

