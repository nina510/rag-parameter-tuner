# 快速开始指南

## API Key 已设置 ✓

API Key 已保存在 `.env` 文件中，启动脚本会自动加载。

## 使用步骤

### 1. 安装依赖（首次使用）

```bash
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
pip install -r requirements.txt
```

### 2. 启动服务器

```bash
./start.sh
```

或者：

```bash
python3 app.py
```

服务器将在 `http://localhost:5000` 启动。

### 3. 打开前端界面

在浏览器中直接打开 `index.html` 文件。

或者使用简单的 HTTP 服务器：

```bash
# 在另一个终端窗口
cd /home/yw28498/specialty-chatbot/RAG/rag-parameter-tuner
python3 -m http.server 8000

# 然后在浏览器访问 http://localhost:8000
```

### 4. 开始使用

1. 在左侧面板输入问题
2. 调整参数（或使用默认值）
3. 点击"生成答案"
4. 可以多次生成不同配置的答案进行对比

## 注意事项

- API Key 存储在 `.env` 文件中，请勿将其提交到版本控制系统
- 确保 `../naivetest/faiss_index` 目录存在
- 首次运行可能需要一些时间加载向量库



