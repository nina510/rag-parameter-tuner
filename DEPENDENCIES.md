# 依赖说明

## Python 依赖

所有 Python 依赖已列在 `requirements.txt` 中，可以通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 代码依赖

### 核心文件

1. **`app.py`** - Flask 后端 API
   - 依赖 `naive_rag.py`（已在同一目录）
   
2. **`naive_rag.py`** - RAG 核心逻辑
   - 依赖 LangChain 相关库（已在 requirements.txt 中）
   - 可选依赖：
     - `load.py` - 如果不存在，相关功能会被禁用
     - `pubmed_service.py` - 如果不存在，PubMed 相关功能会被禁用
     - `config.py` - 如果不存在，会使用默认值

### 可选依赖（可选功能）

`naive_rag.py` 中有一些可选依赖，如果不存在也不会影响核心功能：

- `load_documents` - 文档加载功能（如果不存在，`load_documents = None`）
- `PubMedService` - PubMed 搜索功能（如果不存在，`PubMedService = None`）
- `tree_of_thoughts` - Tree of Thoughts 功能（如果不存在，相关功能会被禁用）

这些依赖通过 `try-except` 处理，不会导致程序崩溃。

## 环境变量

### 必需

- `OPENAI_API_KEY` - OpenAI API 密钥（必需）

### 可选

- `FLASK_DEBUG` - Flask 调试模式（生产环境应设置为 `False`）
- `PORT` - 服务器端口（通常由部署平台自动设置）
- `ALLOWED_ORIGINS` - CORS 允许的域名（用逗号分隔）

## 部署注意事项

1. 确保所有 `requirements.txt` 中的包都已安装
2. 设置 `OPENAI_API_KEY` 环境变量
3. `naive_rag.py` 中的可选依赖如果不存在不会影响核心功能

