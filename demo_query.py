"""
Demo: 问题分解 + 混合检索 + 相关性判断

实现PRD中的部分功能：
1. 问题分解Agent：将用户问题分解为最多5个子检索肯定句
2. 混合检索：BM25 + 语义检索（并行执行）
3. 相关性判断Agent：从文本块中提取相关段落，标注来源，生成子答案
4. 记录输入输出到md文件用于观察
"""
import asyncio
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import jieba
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, SparseVector

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# 配置常量
# ============================================
COLLECTION_NAME = "eye_ana_guide"
PROMPTS_DIR = Path(r"C:\Users\Administrator\Desktop\eyesight_ai_counselor\prompts")
QUERY_LOG_DIR = Path(os.getenv("QUERY_LOG_DIR", "./query_logs"))

# Agent配置
DECOMPOSITION_MODEL = os.getenv("DECOMPOSITION_MODEL", "gpt-4.1")
DECOMPOSITION_TEMPERATURE = float(os.getenv("DECOMPOSITION_TEMPERATURE", "0.3"))
RELEVANCE_MODEL = os.getenv("RELEVANCE_MODEL", "gpt-4.1")
RELEVANCE_TEMPERATURE = float(os.getenv("RELEVANCE_TEMPERATURE", "0.2"))

# 检索配置
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))
BM25_THRESHOLD = float(os.getenv("BM25_THRESHOLD", "2"))
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "3"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.2"))
MAX_SUB_QUERIES = int(os.getenv("MAX_SUB_QUERIES", "5"))

# Embedding配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# LLM配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")


# ============================================
# 数据结构定义
# ============================================
class SubQuery(BaseModel):
    """子检索结构"""
    id: str
    statement: str
    focus: str


class DecompositionResult(BaseModel):
    """问题分解输出"""
    sub_queries: list[SubQuery]
    reasoning: str


class RelevantExtract(BaseModel):
    """相关段落提取"""
    source_file: str
    excerpt: str
    relevance: Literal["high", "medium", "low"]


class RelevanceJudgeResult(BaseModel):
    """相关性判断输出"""
    sub_query_id: str
    sub_query_statement: str
    relevant_extracts: list[RelevantExtract]
    sub_answer: str
    confidence: Literal["high", "medium", "low"]
    missing_aspects: list[str]


class RetrievedChunk(BaseModel):
    """检索到的文本块"""
    source_file: str
    score: float
    content: str


# ============================================
# 核心服务类
# ============================================
class DemoQueryService:
    """Demo查询服务"""

    def __init__(self):
        # 初始化Qdrant客户端
        self.qdrant_client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
        )

        # 初始化OpenAI客户端（用于Embedding）
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        )

        # 加载提示词
        self.decomposition_system_prompt = self._load_prompt("decomposition_system_prompt.txt")
        self.decomposition_user_prompt_template = self._load_prompt("decomposition_user_prompt.txt")
        self.relevance_system_prompt = self._load_prompt("relevance_judge_system_prompt.txt")
        self.relevance_user_prompt_template = self._load_prompt("relevance_judge_user_prompt.txt")

        # 创建Agent
        self.decomposition_agent = self._create_decomposition_agent()
        self.relevance_agent = self._create_relevance_agent()

        # 确保日志目录存在
        QUERY_LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("DemoQueryService 初始化完成")

    def _load_prompt(self, filename: str) -> str:
        """加载提示词文件"""
        filepath = PROMPTS_DIR / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _create_decomposition_agent(self) -> Agent:
        """创建问题分解Agent"""
        provider = OpenAIProvider(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY
        )
        model = OpenAIChatModel(
            DECOMPOSITION_MODEL,
            provider=provider
        )
        return Agent(
            model=model,
            system_prompt=self.decomposition_system_prompt,
            output_type=DecompositionResult
        )

    def _create_relevance_agent(self) -> Agent:
        """创建相关性判断Agent"""
        provider = OpenAIProvider(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY
        )
        model = OpenAIChatModel(
            RELEVANCE_MODEL,
            provider=provider
        )
        return Agent(
            model=model,
            system_prompt=self.relevance_system_prompt,
            output_type=RelevanceJudgeResult
        )

    def build_sparse_vector(self, text: str) -> dict:
        """构建BM25稀疏向量"""
        if not text.strip():
            return {"indices": [], "values": []}

        tokens = list(jieba.cut(text.strip()))
        token_freq = {}
        for token in tokens:
            if token.strip() and len(token.strip()) > 0:
                token_freq[token] = token_freq.get(token, 0) + 1

        indices = []
        values = []
        for token, freq in token_freq.items():
            token_bytes = token.encode('utf-8')
            token_hash = int(hashlib.md5(token_bytes).hexdigest()[:8], 16)
            indices.append(token_hash)
            values.append(float(freq))

        return {"indices": indices, "values": values}

    async def embed_text(self, text: str) -> list[float]:
        """获取文本的向量嵌入"""
        response = await self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.strip()
        )
        return response.data[0].embedding

    async def bm25_search(self, query: str) -> list[RetrievedChunk]:
        """BM25检索"""
        sparse_vector = self.build_sparse_vector(query)
        if not sparse_vector["indices"]:
            return []

        results = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            ),
            using="bm25",
            limit=BM25_TOP_K,
            score_threshold=BM25_THRESHOLD,
            with_payload=True
        )

        chunks = []
        for point in results.points:
            chunks.append(RetrievedChunk(
                source_file=point.payload.get("filename", "unknown"),
                score=point.score,
                content=point.payload.get("content", "")
            ))
        return chunks

    async def semantic_search(self, query: str) -> list[RetrievedChunk]:
        """语义检索"""
        embedding = await self.embed_text(query)

        results = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            using="dense",
            limit=SEMANTIC_TOP_K,
            score_threshold=SEMANTIC_THRESHOLD,
            with_payload=True
        )

        chunks = []
        for point in results.points:
            chunks.append(RetrievedChunk(
                source_file=point.payload.get("filename", "unknown"),
                score=point.score,
                content=point.payload.get("content", "")
            ))
        return chunks

    async def hybrid_search(self, query: str) -> list[RetrievedChunk]:
        """混合检索：并行执行BM25和语义检索，然后合并去重"""
        bm25_task = self.bm25_search(query)
        semantic_task = self.semantic_search(query)

        bm25_results, semantic_results = await asyncio.gather(bm25_task, semantic_task)

        # 合并去重（基于content）
        seen_contents = set()
        merged = []

        for chunk in bm25_results + semantic_results:
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(chunk)

        return merged

    async def decompose_question(self, question: str) -> DecompositionResult:
        """问题分解"""
        user_prompt = self.decomposition_user_prompt_template.replace(
            "{{original_query}}", question
        )
        result = await self.decomposition_agent.run(user_prompt)
        return result.output

    async def judge_relevance(
        self,
        sub_query: SubQuery,
        chunks: list[RetrievedChunk]
    ) -> RelevanceJudgeResult:
        """相关性判断"""
        # 构建检索结果文本
        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            chunks_text += f"""---
**文本块 {idx}**
- 来源文件: {chunk.source_file}
- 相似度分数: {chunk.score:.4f}
- 内容:
\"\"\"
{chunk.content}
\"\"\"
"""

        # 替换模板变量
        user_prompt = self.relevance_user_prompt_template
        user_prompt = user_prompt.replace("{{sub_query_id}}", sub_query.id)
        user_prompt = user_prompt.replace("{{sub_query_statement}}", sub_query.statement)
        user_prompt = user_prompt.replace("{{sub_query_focus}}", sub_query.focus)

        # 替换Handlebars循环为实际内容
        # 找到 {{#each retrieved_chunks}} 和 {{/each}} 之间的内容并替换
        user_prompt = user_prompt.split("{{#each retrieved_chunks}}")[0] + chunks_text + \
                      user_prompt.split("{{/each}}")[-1]

        result = await self.relevance_agent.run(user_prompt)
        return result.output

    async def process_query(self, question: str) -> dict:
        """处理完整查询流程"""
        logger.info(f"开始处理问题: {question}")

        # 1. 问题分解
        logger.info("步骤1: 问题分解")
        decomposition = await self.decompose_question(question)
        logger.info(f"分解为 {len(decomposition.sub_queries)} 个子检索")

        # 2. 并行执行混合检索
        logger.info("步骤2: 并行执行混合检索")
        search_tasks = [
            self.hybrid_search(sq.statement)
            for sq in decomposition.sub_queries
        ]
        all_chunks = await asyncio.gather(*search_tasks)

        # 3. 并行执行相关性判断
        logger.info("步骤3: 并行执行相关性判断")
        relevance_tasks = [
            self.judge_relevance(sq, chunks)
            for sq, chunks in zip(decomposition.sub_queries, all_chunks)
        ]
        all_relevance = await asyncio.gather(*relevance_tasks)

        # 4. 构建结果
        result = {
            "original_question": question,
            "decomposition": decomposition,
            "search_results": [
                {
                    "sub_query": sq.model_dump(),
                    "retrieved_chunks": [c.model_dump() for c in chunks],
                    "relevance_judge": rel.model_dump()
                }
                for sq, chunks, rel in zip(decomposition.sub_queries, all_chunks, all_relevance)
            ]
        }

        # 5. 写入日志
        self._write_log(result)

        return result

    def _write_log(self, result: dict):
        """写入查询日志到md文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = QUERY_LOG_DIR / f"query_{timestamp}.md"

        content = f"""# 查询日志

**时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 原始问题

{result["original_question"]}

## 问题分解

**分解理由**: {result["decomposition"].reasoning}

### 子检索列表

"""
        for sq in result["decomposition"].sub_queries:
            content += f"""- **{sq.id}**: {sq.statement}
  - 聚焦点: {sq.focus}

"""

        content += "## 检索与判断结果\n\n"

        for item in result["search_results"]:
            sq = item["sub_query"]
            chunks = item["retrieved_chunks"]
            rel = item["relevance_judge"]

            content += f"""### {sq["id"]}: {sq["statement"]}

**聚焦点**: {sq["focus"]}

#### 检索到的文本块 ({len(chunks)}个)

"""
            for idx, chunk in enumerate(chunks):
                content += f"""**文本块 {idx + 1}** (来源: {chunk["source_file"]}, 分数: {chunk["score"]:.4f})

```
{chunk["content"][:500]}{"..." if len(chunk["content"]) > 500 else ""}
```

"""

            content += f"""#### 相关性判断结果

**子答案**: {rel["sub_answer"]}

**置信度**: {rel["confidence"]}

**提取的相关段落**:

"""
            for extract in rel["relevant_extracts"]:
                content += f"""- **来源**: {extract["source_file"]} (相关性: {extract["relevance"]})
  > {extract["excerpt"][:300]}{"..." if len(extract["excerpt"]) > 300 else ""}

"""

            if rel["missing_aspects"]:
                content += "**缺失的知识点**:\n"
                for aspect in rel["missing_aspects"]:
                    content += f"- {aspect}\n"

            content += "\n---\n\n"

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"日志已写入: {log_file}")

    async def close(self):
        """关闭连接"""
        await self.qdrant_client.close()
        await self.openai_client.close()


# ============================================
# 主函数
# ============================================
async def main():
    """主函数"""
    service = DemoQueryService()

    try:
        # 测试问题
        test_question = "OK镜适合多大的孩子戴？有什么注意事项？"

        print(f"\n{'='*60}")
        print(f"测试问题: {test_question}")
        print(f"{'='*60}\n")

        result = await service.process_query(test_question)

        # 打印结果摘要
        print("\n" + "="*60)
        print("处理结果摘要")
        print("="*60)

        print(f"\n原始问题: {result['original_question']}")
        print(f"分解理由: {result['decomposition'].reasoning}")

        print(f"\n子检索数量: {len(result['decomposition'].sub_queries)}")
        for sq in result['decomposition'].sub_queries:
            print(f"  - {sq.id}: {sq.statement} (聚焦: {sq.focus})")

        print("\n各子检索的判断结果:")
        for item in result['search_results']:
            rel = item['relevance_judge']
            print(f"\n  [{rel['sub_query_id']}] 置信度: {rel['confidence']}")
            print(f"  子答案: {rel['sub_answer'][:100]}...")
            print(f"  提取段落数: {len(rel['relevant_extracts'])}")

        print(f"\n详细日志已保存到 {QUERY_LOG_DIR} 目录")

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
