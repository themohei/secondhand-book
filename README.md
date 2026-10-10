# 📚 校园二手书助手（大学生二手书交易智能体）

一个用 **Python + DeepSeek + Streamlit** 搭建的校园二手书交易智能体，能跑、能演示、能写进课程报告。核心能力：**拍照识书、智能估价、找书、教材核验、议价助手、交易安全、文案生成、智能客服**。

---

## 一、技术选型（为什么这么选）

| 层 | 选型 | 为什么 |
|---|---|---|
| 大模型 | **DeepSeek**（deepseek-chat） | 极便宜（跑完整个演示几分钱），OpenAI 兼容接口，中文好 |
| 智能体逻辑 | 纯 `openai` SDK（没用 LangChain） | 逻辑简单清晰，少一层依赖，课堂上更好讲原理 |
| 界面 | **Streamlit** | 几十行代码就能出网页，适合 Demo |
| 知识库 | 本地 JSON（书库 + FAQ） | 零部署，改数据直接改文件 |
| 检索 | 字符 2-gram 重叠打分（轻量 RAG） | 不装分词库和向量库，原理易懂 |

> 想换免费模型（智谱 GLM-4-Flash 等）或加向量库（Chroma）？看文末「进阶」。

---

## 二、从零开始的完整步骤

### 第 0 步：装软件（三选二必装）

1. **Python 3.10~3.12**：去 <https://www.python.org/downloads/> 下载安装。
   ⚠️ 安装第一屏**务必勾选「Add Python to PATH」**，否则后面命令会报「python 不是内部或外部命令」。
2. **VS Code**：去 <https://code.visualstudio.com/> 下载，装完在左侧「扩展」里搜索安装 **Python** 扩展。
3. **Git**（可选）：去 <https://git-scm.com/> 下载，一路下一步。

装完后按 `Win + R` 输入 `powershell` 回车，在终端里验证：

```powershell
python --version
# 应输出类似：Python 3.12.x
```

### 第 1 步：进入项目目录

在 PowerShell 里执行（把路径换成你实际的）：

```powershell
cd F:\Deepseek_Harness\secondhand-book-agent
```

### 第 2 步：创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 如果 `Activate.ps1` 报「禁止运行脚本」，先执行一句：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> 然后重新运行 `.\\.venv\Scripts\Activate.ps1`。
> 激活成功后，命令行前面会出现 `(.venv)`。

### 第 3 步：注册 DeepSeek，拿到 API Key

1. 打开 <https://platform.deepseek.com> 注册（手机号即可）。
2. 左侧「API Keys」→「创建 API Key」，复制那串 `sk-...`。
3. 新用户一般会送一些免费额度；之后按量计费，**非常便宜**。

### 第 4 步：把 Key 填进 `.env`

把项目里的 `.env.example` 复制一份，改名为 `.env`，用记事本或 VS Code 打开，把 Key 填进去：

```text
DEEPSEEK_API_KEY=sk-你的真实key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_VISION_MODEL=deepseek-flash
```

保存。

### 第 5 步：运行

```powershell
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，看到页面就算成功。

---

## 三、怎么演示（照着这 7 个 Tab 走）

1. **📸 拍照估价**：上传一本书的封面/封底照片 → 选品相 → 点「拍照识别并估价」→ 自动识别书名版次并给估价；也可直接输入 ISBN/书名估价。
2. **🔍 找书**：输入书名/作者/ISBN/课程搜索，或按专业浏览书库。
3. **🎓 教材核验**：输入课程名（如「数据结构」）→ 列出多本常用教材 + ISBN + 参考价。
4. **🤝 议价助手**：选中一本书 → 填卖家报价 → 得到是否划算 + 还价目标价 + 议价话术。
5. **🛡️ 交易安全**：描述交易场景 → 检测私下转账/超低价/陌生链接等风险并给避坑提醒。
6. **✍️ 发布文案**：选书 → 选品相 → 一键生成可发二手群的出售文案。
7. **💬 智能客服**：聊天框输入「八五新是什么意思？」等 → AI 检索 FAQ 并回答。

---

## 四、代码结构（写报告时对照）

```
secondhand-book-agent/
├── app.py           # Streamlit 界面（7 个 Tab）
├── agent.py         # 智能体核心：模型/视觉调用 + 书库查询 + 检索 + 议价 + 安全
├── data/
│   ├── books.json   # 书库数据（80 本书，含课程/专业映射）
│   └── faq.json     # 常见问题知识库（16 条）
├── requirements.txt # 依赖清单
├── .env             # API Key 配置（已 gitignore，需自己填）
└── README.md        # 本文档
```

- **估价** = 规则算区间（品相折价系数）× 大模型写说明 → 体现「数据 + 大模型」结合。
- **客服** = 检索 FAQ（字符 2-gram 打分）→ 拼进提示词让模型作答 → 就是 PPT 里的「RAG 检索增强生成」。
- **版本核验** = 书库的 `course` 字段映射，正是你 PPT 里「版本信息自动核验」的最小实现。

---

## 五、常见报错排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `python 不是内部或外部命令` | 装 Python 没勾 Add to PATH | 重装 Python 并勾选，或重开终端 |
| `禁止运行脚本` | PowerShell 执行策略 | 见第 2 步里的 `Set-ExecutionPolicy` |
| 页面提示「智能服务暂未连接」 | `.env` 里 Key 为空或填错 | 填 Key 后重启（Ctrl+C 停止，再 `streamlit run app.py`） |
| `401 / AuthenticationError` | Key 填错或额度用完 | 检查 Key 是否复制完整；到控制台看余额 |
| `Connection error` | 没联网 / 网络被墙 | 检查网络，校园网有时需开代理 |

---

## 六、进阶（可选，做完基础版再说）

1. **换免费模型**：智谱 GLM-4-Flash 免费。改 `.env`：
   ```text
   DEEPSEEK_BASE_URL=https://open.bigmodel.cn/api/paas/v4
   DEEPSEEK_MODEL=glm-4-flash
   ```
   （Key 填智谱的，代码一行不用改——这就是用 OpenAI 兼容接口的好处。）
2. **换视觉模型**：`.env` 里 `DEEPSEEK_VISION_MODEL` 换成其他支持视觉的 OpenAI 兼容模型即可。
3. **换成真正的向量库**：`pip install chromadb`，把 FAQ 的 2-gram 检索换成向量相似度检索。
4. **扩充书库**：往 `data/books.json` 里加书（字段照着现有格式复制），就能估更多书。

---

## 七、对应到你 PPT 的一页

- 技术平台页的「LangChain + Pinecone/Milvus」建议改成上面这张表：**DeepSeek + Streamlit + 本地 JSON 书库 + 轻量 RAG**，理由是「MVP 阶段用最小成本验证核心流程」，显得你有工程判断力，而不是堆名词。
- 演示时加一页「系统演示截图」，把四个 Tab 各截一张图放进去。
