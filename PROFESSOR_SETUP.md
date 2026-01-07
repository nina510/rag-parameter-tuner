# 教授电脑本地运行指南

## 一、需要提前安装的软件

### 1. Python 3.9 或更高版本（必需）

**Windows 用户：**
1. 访问 https://www.python.org/downloads/
2. 下载 Python 3.9+ 安装包
3. 安装时 **务必勾选 "Add Python to PATH"**（非常重要！）
4. 安装完成后，打开 PowerShell，输入 `python --version` 确认安装成功

**Mac 用户：**
```bash
# 如果已安装 Homebrew
brew install python

# 或从官网下载：https://www.python.org/downloads/
```

### 2. 不需要 IDE

只需要：
- 浏览器（Chrome / Edge / Firefox / Safari）
- 终端 / 命令行（Windows: PowerShell 或 CMD；Mac: Terminal）

---

## 二、需要拷贝的文件

将整个 `rag-parameter-tuner` 文件夹拷贝到教授电脑上。

### 文件夹结构应该包含：
```
rag-parameter-tuner/
├── app.py              # 后端主程序
├── naive_rag.py        # RAG 核心逻辑
├── load.py             # 文档加载工具
├── index.html          # 前端界面
├── requirements.txt    # Python 依赖
├── .env                # API 密钥配置（需要创建）
└── corpus/             # 文档语料库
    ├── csv/
    │   └── LongCovid.csv
    └── txt/
        └── LongCovid/
            └── (8个 .txt 文件)
```

### 重要：需要创建 .env 文件

**⚠️ 注意：`.env` 文件名以点开头，macOS Finder 不允许直接创建，需要使用以下方法：**

#### Windows 用户：
直接创建文件即可：
1. 在文件夹中右键 → 新建 → 文本文档
2. 重命名为 `.env`（如果提示扩展名，选择"是"）
3. 用记事本打开，输入：
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

#### Mac 用户：
需要使用终端创建：

1. **打开 Terminal**（按 `Cmd + Space`，搜索 "Terminal"）

2. **进入项目文件夹**
   ```bash
   cd /path/to/rag-parameter-tuner
   ```
   （可以直接把文件夹拖到 Terminal 窗口，会自动输入路径）

3. **创建 .env 文件**
   ```bash
   echo "OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx" > .env
   ```
   （将 `sk-proj-xxxx...` 替换为实际的 OpenAI API 密钥）

4. **验证文件已创建**
   ```bash
   ls -la .env
   ```
   如果看到 `.env` 文件，说明创建成功。

**或者使用文本编辑器：**
1. 打开 TextEdit（文本编辑）
2. 输入：`OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx`
3. 保存时，文件格式选择 "纯文本"
4. 保存为 `env`（不带点）
5. 在终端运行：`mv env .env`（将文件重命名为 .env）

---

## 三、首次运行步骤

### Windows 用户：

1. **打开 PowerShell**
   - 按 `Win + X`，选择 "Windows PowerShell"

2. **进入项目文件夹**
   ```powershell
   cd C:\path\to\rag-parameter-tuner
   ```
   （将路径替换为实际路径）

3. **安装依赖**
   ```powershell
   pip install -r requirements.txt
   ```
   这一步只需要在首次运行时执行，之后不需要重复。

4. **启动应用**
   ```powershell
   python app.py
   ```

5. **打开浏览器**
   在浏览器地址栏输入：
   ```
   http://localhost:5000
   ```

### Mac 用户：

1. **打开 Terminal**
   - 按 `Cmd + Space`，搜索 "Terminal"

2. **进入项目文件夹**
   ```bash
   cd /path/to/rag-parameter-tuner
   ```

3. **安装依赖**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **启动应用**
   ```bash
   python3 app.py
   ```

5. **打开浏览器**
   访问 `http://localhost:5000`

---

## 四、每次使用步骤（安装依赖后）

1. 打开终端
2. 进入项目文件夹：`cd /path/to/rag-parameter-tuner`
3. 启动：`python app.py`（Windows）或 `python3 app.py`（Mac）
4. 浏览器访问：`http://localhost:5000`
5. 使用完毕后，在终端按 `Ctrl + C` 停止服务

---

## 五、常见问题

### Q: 出现 "python not found" 错误
A: Python 未正确安装或未添加到 PATH。请重新安装 Python，确保勾选 "Add Python to PATH"。

### Q: 出现 "pip not found" 错误
A: 尝试使用 `python -m pip install -r requirements.txt`

### Q: 出现 "OPENAI_API_KEY not set" 错误
A: 确保 `.env` 文件存在且包含正确的 API 密钥。

### Q: 浏览器显示 "无法连接"
A: 确保终端中显示 "Running on http://0.0.0.0:5000"，且没有报错。

### Q: 依赖安装很慢或失败
A: 可以尝试使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 六、快速启动脚本（可选）

### Windows：创建 `start.bat` 文件

```batch
@echo off
cd /d "%~dp0"
python app.py
pause
```

双击 `start.bat` 即可启动。

### Mac/Linux：创建 `start.sh` 文件

```bash
#!/bin/bash
cd "$(dirname "$0")"
python3 app.py
```

运行前需要赋予执行权限：`chmod +x start.sh`
然后双击或运行 `./start.sh` 即可启动。

