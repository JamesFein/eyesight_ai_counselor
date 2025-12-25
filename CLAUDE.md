# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

眼科近视防控AI咨询系统，使用RAG（检索增强生成）技术为用户提供基于权威医学文档的问答服务。

## 常用命令

### 环境配置
```bash
# 创建虚拟环境（首次运行）
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据导入
```bash
# 将data目录中的权威文档导入Qdrant向量数据库
python ingest.py
```

### 问答测试
```bash
# 启动交互式问答系统
python query.py
```

### Qdrant数据库
```bash
# 启动Qdrant（需要Docker）
docker run -p 6333:6333 -p 6334:6334 \
    -v qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

## 代码架构

### 核心模块

#### 1. 数据导入流程 (ingest.py)
- **DataIngestion类**：负责文档摄取和向量化
- **文本分块策略**：使用RecursiveChunker按Markdown标题层级递归分割
  - 优先级：一级标题 → 二级标题 → 三级标题 → 段落 → 句子 → Token
  - 默认chunk_size: 1024 tokens，overlap: 0
- **向量存储**：同时存储密集向量(dense)和稀疏向量(bm25)到Qdrant
  - dense: OpenAI text-embedding-3-small (1536维)
  - bm25: 基于jieba分词的哈希稀疏向量（用于关键词匹配）

#### 2. 问答系统流程 (query.py)
- **QueryService类**：负责问答和检索
- **两阶段Agent架构**：
  1. **问题改写Agent** (rewrite_agent)：将简短问题改写为包含上下文的完整问题
  2. **RAG回答Agent** (answer_agent)：基于检索文档生成答案
- **混合检索**：并行执行BM25和语义检索，然后去重合并
  - BM25检索：适合关键词匹配（医学术语、专有名词）
  - 语义检索：适合理解语义相似的问题

### 数据目录结构

- **data/**：权威医学文档（9个.md文件）
  - 包含近视管理白皮书、专家共识、防治指南等
  - 这些文档是知识库的唯一来源
- **docs/**：项目文档
  - 文本分块策略说明
  - 文档结构化改写提示词
- **demo_report/**：示例用户数据（测试用）

### 配置管理

所有配置通过`.env`文件管理，主要分为：
- **LLM配置**：用于对话生成（LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE等）
- **问题改写配置**：专门用于问题改写的LLM配置（QUERY_REWRITE_*）
- **Embedding配置**：用于向量化（EMBEDDING_API_KEY, EMBEDDING_MODEL等）
- **Qdrant配置**：向量数据库连接（QDRANT_HOST, QDRANT_PORT等）
- **检索参数**：BM25_TOP_K, SEMANTIC_TOP_K, 阈值等

## Windows环境特殊要求

### 文件路径规范（重要！）
- **必须使用Windows绝对路径**：`C:\Users\Administrator\Desktop\eyesight_ai_counselor\file.md`
- **禁止使用相对路径**：`file.md`
- **禁止使用正斜杠**：`C:/Users/Administrator/...`

### 命令兼容性
- 优先使用专用工具：Glob（文件搜索）、Grep（内容搜索）、Read/Write/Edit（文件操作）
- 避免使用Linux特有命令：`tree`、`grep`、`find`
- 使用跨平台命令：`ls`（Git Bash环境）

### Edit工具使用流程
1. 先用Read工具读取目标区域
2. 精确复制要替换的文本（包括所有空格、标点、换行）
3. 确认匹配后再调用Edit

## MCP服务器

项目配置了以下MCP服务器（.mcp.json）：
- Context 7：库文档查询
- pydantic-ai Docs：Pydantic-AI框架文档
- qdrant Docs：Qdrant向量数据库文档
- chonkie Docs：Chonkie文本分块库文档
- chrome-devtools：浏览器调试工具

## 技术栈

- **Web框架**：FastAPI + Uvicorn
- **AI框架**：Pydantic-AI (Agent框架)
- **向量数据库**：Qdrant
- **文本处理**：Chonkie (分块), Jieba (中文分词)
- **Embedding**：OpenAI text-embedding-3-small
- **LLM**：OpenAI GPT-4o / GPT-4o-mini (可配置)

## 开发注意事项

### 数据导入
- ingest.py会自动跳过已存在的文件（根据filename字段）
- 修改文档后需要清空collection重新导入，或删除特定文件的points
- 分块参数调整需要重新导入数据

### 问答系统
- query.py默认路径引用了另一个项目的prompts目录（需要调整）
  - 当前：`C:\Users\Administrator\Desktop\fast-gzmdrw-chat\backend\app\prompts`
  - 需要：在本项目创建prompts目录
- 提示词文件：
  - rewrite_question_system_prompt.txt
  - rewrite_question_user_prompt.txt
  - rag_agent_system_prompt.txt

### 向量检索优化
- 调整BM25_TOP_K和SEMANTIC_TOP_K控制召回数量
- 调整SEMANTIC_THRESHOLD过滤低相关度结果
- BM25适合精确关键词匹配，语义检索适合理解相似问题

### Collection管理
Collection名称：`eye_ana_guide`
- 包含dense向量（1536维）和bm25稀疏向量
- Payload字段：content, filename, chunk_index, total_chunks, chunk_size
