# RAG 参数调节工具

这是一个基于 `naive_rag.py` 的交互式工具，允许您实时调节 RAG 系统的参数并对比不同配置生成的答案。

## 功能特性

- ✅ **实时参数调节**：
  - System Prompt（可自定义或使用默认）
  - Chunk Size（默认: 1200）
  - Chunk Overlap（默认: 600）
  - 模型选择（GPT-4o, GPT-4, GPT-4 Turbo, GPT-3.5 Turbo）
  - 检索器选择（OpenAI Embedding, BM25, Contriever, SPECTER, MedCPT）
  - Top-K（检索数量，默认: 25）

- ✅ **答案对比**：可以生成多个不同参数配置的答案，并并排对比

- ✅ **默认设置**：保持与 `naive_rag.py` 相同的默认参数

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量：
```bash
export OPENAI_API_KEY='your-api-key'
```

## 使用方法

### 方法一：使用启动脚本（推荐）

```bash
./start.sh
```

脚本会自动检查依赖和环境变量，然后启动服务器。

### 方法二：手动启动

1. **安装依赖**：
```bash
pip install -r requirements.txt
```

2. **设置环境变量**：
```bash
export OPENAI_API_KEY='your-api-key'
```

3. **启动后端服务器**：
```bash
python app.py
```

服务器将在 `http://localhost:5000` 启动。

4. **打开前端界面**：
   
   方式 A：直接在浏览器中打开 `index.html` 文件（推荐）
   
   方式 B：使用简单的 HTTP 服务器：
   ```bash
   # Python 3
   python -m http.server 8000
   
   # 然后在浏览器访问 http://localhost:8000
   ```

### 使用工具

1. **输入问题**：在左侧面板的"问题"文本框中输入您的问题

2. **调整参数**（可选）：
   - **System Prompt**：留空使用默认的 Long COVID 专用 prompt，或输入自定义 prompt
   - **Chunk Size**：文档分块大小（默认: 1200）
   - **Chunk Overlap**：分块重叠大小（默认: 600）
   - **模型**：选择使用的 GPT 模型（默认: GPT-4o）
   - **检索器**：选择检索器类型（默认: OpenAI Embedding）
   - **Top K**：检索的文档数量（默认: 25）

3. **生成答案**：点击"生成答案"按钮

4. **对比结果**：
   - 可以多次生成不同配置的答案
   - 所有答案会并排显示在右侧对比区域
   - 每个答案卡片显示其参数配置

5. **清空结果**：点击"清空结果"按钮清除所有已生成的结果

## API 端点

### POST `/api/generate`
生成答案

**请求体：**
```json
{
  "question": "您的问题",
  "system_prompt": "自定义 system prompt（可选）",
  "chunk_size": 1200,
  "chunk_overlap": 600,
  "model": "gpt-4o",
  "retriever_name": "OpenAIEmbedding",
  "top_k": 25
}
```

**响应：**
```json
{
  "answer": "生成的答案",
  "chunks": [...],
  "metadata": {
    "model": "gpt-4o",
    "chunk_size": 1200,
    "chunk_overlap": 600,
    "num_chunks": 25,
    "retriever_name": "OpenAIEmbedding"
  }
}
```

### GET `/api/defaults`
获取默认参数

### GET `/api/health`
健康检查

## 注意事项

1. **FAISS 索引**：确保 `../naivetest/faiss_index` 目录存在且包含有效的 FAISS 索引
2. **API Key**：确保已设置 `OPENAI_API_KEY` 环境变量
3. **首次运行**：首次运行可能需要一些时间来加载向量库
4. **System Prompt**：留空将使用默认的 Long COVID 专用 prompt
5. **Chunk Size/Overlap**：这些参数主要用于记录，因为索引已经构建完成。如果需要使用不同的 chunk 参数，需要重新构建索引
6. **检索器选择**：
   - **OpenAI Embedding**：使用 FAISS 向量库（推荐，最快）
   - **BM25/Contriever/SPECTER/MedCPT**：使用 iMedRAG 检索系统（需要 iMedRAG 相关文件）
7. **跨域问题**：如果前端和后端不在同一端口，可能需要配置 CORS（已在代码中启用）

## 文件结构

```
rag-parameter-tuner/
├── app.py              # Flask 后端 API
├── index.html          # 前端界面
├── requirements.txt    # Python 依赖
├── start.sh           # 启动脚本
└── README.md          # 本文件
```

## 技术细节

### 后端架构

- **Flask**：轻量级 Web 框架
- **CORS**：支持跨域请求
- **向量库缓存**：首次加载后缓存，提高性能

### 前端特性

- **响应式设计**：支持不同屏幕尺寸
- **实时对比**：并排显示多个答案
- **Markdown 渲染**：自动将 Markdown 格式转换为 HTML
- **参数记录**：每个答案卡片显示其参数配置

### 参数说明

- **Chunk Size/Overlap**：虽然当前索引已构建，这些参数会被记录在元数据中，方便对比分析
- **System Prompt**：完全自定义，可以覆盖默认的 Long COVID prompt
- **模型选择**：支持所有 OpenAI 的 GPT 模型
- **检索器**：支持 OpenAI Embedding 和 iMedRAG 检索器

