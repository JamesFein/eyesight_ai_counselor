"""
问答脚本
对用户输入的问题进行改写和检索，使用BM25和语义混合检索
"""
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from pathlib import Path

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import SparseVector
from openai import AsyncOpenAI
from pydantic_ai import Agent
import jieba
import hashlib

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置常量
COLLECTION_NAME = "eye_ana_guide"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "5"))
BM25_THRESHOLD = float(os.getenv("BM25_THRESHOLD", "0.0"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.5"))
PROMPTS_DIR = Path(r"C:\Users\Administrator\Desktop\fast-gzmdrw-chat\backend\app\prompts")


class QueryService:
    """问答服务类"""
    
    def __init__(self):
        # 初始化 Qdrant 客户端
        self.qdrant_client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
        )
        
        # 初始化 OpenAI 客户端（用于embedding）
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        )
        
        # 加载提示词
        self.rewrite_system_prompt = self._load_prompt("rewrite_question_system_prompt.txt")
        self.rewrite_user_prompt_template = self._load_prompt("rewrite_question_user_prompt.txt")
        self.rag_system_prompt = self._load_prompt("rag_agent_system_prompt.txt")
        
        # 配置问题改写 Agent（使用LLM配置）
        os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY")
        os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        
        self.rewrite_agent = Agent(
            model=f'openai:{os.getenv("QUERY_REWRITE_LLM_MODEL", "gpt-4o")}',
            result_type=str,
            system_prompt=self.rewrite_system_prompt,
            model_settings={
                'temperature': float(os.getenv("QUERY_REWRITE_LLM_TEMPERATURE", "0.1")),
                'max_tokens': int(os.getenv("QUERY_REWRITE_LLM_MAX_TOKENS", "300")),
            }
        )
        
        # 配置回答 Agent（使用LLM配置）
        self.answer_agent = Agent(
            model=f'openai:{os.getenv("LLM_MODEL", "gpt-4o-mini")}',
            result_type=str,
            system_prompt=self.rag_system_prompt,
            model_settings={
                'temperature': float(os.getenv("LLM_TEMPERATURE", "0.7")),
                'max_tokens': int(os.getenv("LLM_MAX_TOKENS", "2000")),
            }
        )
        
        logger.info("问答服务初始化完成")
    
    def _load_prompt(self, filename: str) -> str:
        """加载提示词文件"""
        prompt_path = PROMPTS_DIR / filename
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    async def rewrite_question(self, question: str) -> str:
        """改写问题（简化版，无历史记录）"""
        logger.info(f"\n{'='*60}")
        logger.info(f"原始问题: {question}")
        
        # 构建用户提示词（无历史记录）
        user_prompt = self.rewrite_user_prompt_template.format(
            history_text="(无历史记录)",
            question=question
        )
        
        try:
            result = await self.rewrite_agent.run(user_prompt=user_prompt)
            rewritten = result.data.strip()
            logger.info(f"改写问题: {rewritten}")
            logger.info(f"{'='*60}\n")
            return rewritten
        except Exception as e:
            logger.error(f"问题改写失败: {e}，使用原始问题")
            logger.info(f"{'='*60}\n")
            return question
    
    def build_sparse_vector(self, text: str) -> Dict[str, Any]:
        """构建 BM25 稀疏向量"""
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
    
    async def embed_text(self, text: str) -> List[float]:
        """获取文本的向量嵌入"""
        response = await self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.strip()
        )
        return response.data[0].embedding
    
    async def bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """BM25 检索"""
        sparse_vector = self.build_sparse_vector(query)
        
        search_results = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            ),
            using="bm25",
            limit=BM25_TOP_K
        )
        
        results = []
        for result in search_results.points:
            if result.score >= BM25_THRESHOLD:
                results.append({
                    "content": result.payload.get("content", ""),
                    "score": result.score,
                    "filename": result.payload.get("filename", ""),
                    "chunk_index": result.payload.get("chunk_index", 0)
                })
        
        return results
    
    async def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """语义检索"""
        query_vector = await self.embed_text(query)
        
        search_results = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="dense",
            limit=SEMANTIC_TOP_K,
            score_threshold=SEMANTIC_THRESHOLD
        )
        
        results = []
        for result in search_results.points:
            results.append({
                "content": result.payload.get("content", ""),
                "score": result.score,
                "filename": result.payload.get("filename", ""),
                "chunk_index": result.payload.get("chunk_index", 0)
            })
        
        return results
    
    async def hybrid_search(self, query: str) -> Dict[str, Any]:
        """混合检索"""
        logger.info("🔍 开始混合检索...")
        
        # 并行执行BM25和语义检索
        bm25_results, semantic_results = await asyncio.gather(
            self.bm25_search(query),
            self.semantic_search(query)
        )
        
        # 去重合并
        seen_contents = set()
        deduplicated = []
        
        for result in bm25_results:
            content = result["content"]
            if content not in seen_contents:
                seen_contents.add(content)
                deduplicated.append(result)
        
        for result in semantic_results:
            content = result["content"]
            if content not in seen_contents:
                seen_contents.add(content)
                deduplicated.append(result)
        
        return {
            "bm25_results": bm25_results,
            "semantic_results": semantic_results,
            "deduplicated_results": deduplicated
        }
    
    def print_retrieval_results(self, results: Dict[str, Any]):
        """打印检索结果"""
        print("\n" + "="*80)
        print("📊 检索结果")
        print("="*80)
        
        print(f"\n🔤 BM25检索结果 ({len(results['bm25_results'])} 个):")
        for i, result in enumerate(results['bm25_results'], 1):
            print(f"\n[{i}] 相似度分数: {result['score']:.4f}")
            print(f"    来源: {result['filename']} (块 {result['chunk_index']})")
            print(f"    内容: {result['content'][:100]}...")
        
        print(f"\n🎯 语义检索结果 ({len(results['semantic_results'])} 个):")
        for i, result in enumerate(results['semantic_results'], 1):
            print(f"\n[{i}] 相似度分数: {result['score']:.4f}")
            print(f"    来源: {result['filename']} (块 {result['chunk_index']})")
            print(f"    内容: {result['content'][:100]}...")
        
        print(f"\n✅ 去重后的结果 ({len(results['deduplicated_results'])} 个):")
        for i, result in enumerate(results['deduplicated_results'], 1):
            print(f"\n[{i}] 相似度分数: {result['score']:.4f}")
            print(f"    来源: {result['filename']} (块 {result['chunk_index']})")
            print(f"    内容: {result['content'][:100]}...")
        
        print("\n" + "="*80 + "\n")
    
    async def answer_question(self, question: str):
        """回答问题"""
        # 1. 改写问题
        rewritten_question = await self.rewrite_question(question)
        
        # 2. 混合检索
        retrieval_results = await self.hybrid_search(rewritten_question)
        
        # 3. 打印检索结果
        self.print_retrieval_results(retrieval_results)
        
        # 4. 构建上下文
        context_parts = []
        for i, result in enumerate(retrieval_results['deduplicated_results'], 1):
            context_parts.append(
                f"[文档{i}] 来源: {result['filename']}\n{result['content']}"
            )
        context = "\n\n".join(context_parts)
        
        if not context:
            print("❌ 没有找到相关文档，无法回答问题。\n")
            return
        
        # 5. 生成回答
        logger.info("💬 生成回答...")
        user_prompt = f"""参考文档：
{context}

用户问题：{question}

请基于以上参考文档回答用户的问题。"""
        
        try:
            result = await self.answer_agent.run(user_prompt=user_prompt)
            answer = result.data.strip()
            
            print("="*80)
            print("💡 回答")
            print("="*80)
            print(answer)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            print(f"❌ 生成回答失败: {e}\n")
    
    async def close(self):
        """关闭连接"""
        await self.qdrant_client.close()
        await self.openai_client.close()


async def main():
    """主函数"""
    service = QueryService()
    
    try:
        print("\n" + "="*80)
        print("🤖 眼科分析问答系统")
        print("="*80)
        print("输入 'exit' 或 'quit' 退出\n")
        
        while True:
            # 获取用户输入
            question = input("请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit']:
                print("\n再见！👋\n")
                break
            
            # 回答问题
            await service.answer_question(question)
            
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
