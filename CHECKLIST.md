# GitHub 部署检查清单

## ✅ 准备上传的文件

### 核心代码文件
- [x] `app.py` - Flask 后端 API (44KB)
- [x] `index.html` - 前端界面 (71KB)
- [x] `requirements.txt` - Python 依赖
- [x] `start.sh` - 启动脚本
- [x] `.gitignore` - Git 忽略配置

### 文档文件
- [x] `README.md` - 项目说明
- [x] `QUICK_START.md` - 快速开始
- [x] `DEPLOYMENT.md` - 部署指南
- [x] `CHECKLIST.md` - 本检查清单

### 依赖模块（需要确认）
- [ ] `../naivetest/naive_rag.py` - RAG 核心模块
  - 如果在同一仓库，确保已包含
  - 如果不在同一仓库，需要处理路径或复制文件

## ✅ 安全检查

- [x] `.env` 文件已在 .gitignore 中（包含 API 密钥）
- [x] `*.log` 文件已在 .gitignore 中
- [x] `__pycache__/` 已在 .gitignore 中
- [x] 代码中没有硬编码的 API 密钥
- [x] API 密钥通过环境变量读取

## ✅ 代码检查

- [x] 所有导入语句正确
- [x] 路径引用正确（相对路径）
- [x] requirements.txt 包含所有依赖
- [x] 启动脚本权限正确（chmod +x start.sh）

## 📋 部署步骤

### 1. 检查当前状态
```bash
cd RAG/rag-parameter-tuner
git status
```

### 2. 确认要提交的文件
```bash
git add app.py index.html requirements.txt start.sh .gitignore README.md QUICK_START.md DEPLOYMENT.md CHECKLIST.md
```

### 3. 确认排除的文件
```bash
# 这些文件应该被忽略（不应该出现在 git status 中）
# .env
# *.log
# __pycache__/
# app.log
```

### 4. 提交
```bash
git commit -m "Add RAG Parameter Tuner: Interactive tool for tuning RAG parameters"
```

### 5. 推送到 GitHub
```bash
# 如果是新仓库
git remote add origin https://github.com/yourusername/your-repo.git
git branch -M main
git push -u origin main

# 如果是现有仓库
git push origin main
```

## ⚠️ 重要提示

1. **依赖模块路径**
   - `app.py` 依赖 `../naivetest/naive_rag.py`
   - 确保仓库结构正确，或修改导入路径

2. **环境变量**
   - 在 GitHub Actions 中添加 Secrets: `OPENAI_API_KEY`
   - 本地开发使用 `.env` 文件（不提交）

3. **文件大小**
   - FAISS 索引文件可能很大（可选，根据需要）
   - 考虑使用 Git LFS 如果索引文件需要共享

4. **测试**
   - 克隆到新目录测试
   - 确认所有依赖能正常安装
   - 确认服务器能正常启动

## 🔍 验证清单

部署后验证：
- [ ] 仓库中能看到所有文件
- [ ] .gitignore 正确排除了敏感文件
- [ ] README.md 能正确显示
- [ ] requirements.txt 完整
- [ ] 代码中没有敏感信息泄露

