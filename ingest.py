"""
数据导入脚本
将 data 目录中的 Markdown 文件导入到 Qdrant 的 eye_ana_guide collection

采用递归分块策略，按照 Markdown 标题层级从高到低依次尝试分割：
1. 一级标题 `# ` → 按章节分块
2. 二级标题 `## ` → 章节过长时，按子章节分块
3. 三级标题 `### ` → 子章节过长时，按小节分块
4. 段落 `\n\n` → 小节过长时，按段落分块
5. 句子（句号、问号、感叹号）→ 段落过长时，按句子分块
6. Token 级别 → 最后兜底
"""
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, Modifier, SparseVector
from openai import AsyncOpenAI
from chonkie import RecursiveChunker, RecursiveRules, RecursiveLevel
import jieba
import hashlib
import uuid

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置常量
COLLECTION_NAME = "eye_ana_guide"
DATA_DIR = Path(r"C:\Users\Administrator\Desktop\eyesight_ai_counselor\data")
# 中文专业文档，chunk_size 设为 800-1024 tokens
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
# 按标题分块时，各章节相互独立，无需重叠
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))
# 避免产生过小的孤立块
MIN_CHARACTERS_PER_CHUNK = int(os.getenv("MIN_CHARACTERS_PER_CHUNK", "100"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


class DataIngestion:
    """数据导入类"""

    def __init__(self):
        # 初始化 Qdrant 客户端
        self.qdrant_client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
        )

        # 初始化 OpenAI 客户端
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        )

        # 定义 Markdown 递归分块规则
        # 按照文档方案，优先级从高到低：
        # 一级标题 → 二级标题 → 三级标题 → 段落 → 句子 → Token
        markdown_rules = RecursiveRules(
            levels=[
                # 一级标题：按章节分块
                RecursiveLevel(delimiters=["\n# "], include_delim="next"),
                # 二级标题：章节过长时，按子章节分块
                RecursiveLevel(delimiters=["\n## "], include_delim="next"),
                # 三级标题：子章节过长时，按小节分块
                RecursiveLevel(delimiters=["\n### "], include_delim="next"),
                # 段落：小节过长时，按段落分块
                RecursiveLevel(delimiters=["\n\n"], include_delim="prev"),
                # 句子：段落过长时，按句子分块（中英文标点）
                RecursiveLevel(delimiters=["。", "！", "？", ". ", "! ", "? "], include_delim="prev"),
                # Token 级别兜底：按空白字符分割
                RecursiveLevel(whitespace=True)
            ]
        )

        # 初始化 RecursiveChunker 分块器
        self.chunker = RecursiveChunker(
            chunk_size=CHUNK_SIZE,
            rules=markdown_rules,
            min_characters_per_chunk=MIN_CHARACTERS_PER_CHUNK
        )

        logger.info("数据导入服务初始化完成")
        logger.info(f"分块参数: chunk_size={CHUNK_SIZE}, min_characters_per_chunk={MIN_CHARACTERS_PER_CHUNK}")
    
    async def ensure_collection_exists(self):
        """确保 collection 存在"""
        collections = await self.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"创建 collection: {COLLECTION_NAME}")
            await self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        modifier=Modifier.IDF
                    )
                }
            )
            logger.info(f"Collection {COLLECTION_NAME} 创建成功")
        else:
            logger.info(f"Collection {COLLECTION_NAME} 已存在")
    
    def preprocess_markdown(self, content: str) -> str:
        """
        预处理 Markdown 文档，确保格式统一

        预处理步骤：
        1. 统一换行符：将 \\r\\n 和 \\r 统一转换为 \\n
        2. 压缩空行：将 3 个及以上连续换行压缩为 2 个
        3. 修复标题格式：确保 # 后有空格
        4. 去除行首行尾空白：每行 strip 处理
        """
        # 1. 统一换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # 2. 修复标题格式：确保 # 后有空格
        # 匹配行首的 1-6 个 # 号后面紧跟非空格非 # 的字符
        content = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', content, flags=re.MULTILINE)

        # 3. 去除每行首尾空白，但保留空行
        lines = content.split('\n')
        lines = [line.strip() for line in lines]
        content = '\n'.join(lines)

        # 4. 压缩空行：将 3 个及以上连续换行压缩为 2 个
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    def chunk_text(self, text: str, filename: str = "") -> List[Dict[str, Any]]:
        """
        使用 RecursiveChunker 分块文本

        返回包含元数据的 chunk 列表
        """
        # 预处理文档
        text = self.preprocess_markdown(text)

        # 检查文档是否过短
        if len(text) < 500:
            logger.warning(f"⚠️  文档内容过短（< 500 字符），整体作为单个 chunk: {filename}")
            return [{
                "text": text,
                "token_count": len(text),
                "chunk_index": 0,
                "start_index": 0,
                "end_index": len(text)
            }]

        # 检查是否为空
        if not text.strip():
            logger.error(f"❌ 空文件，跳过: {filename}")
            return []

        # 使用 RecursiveChunker 分块
        chonks = self.chunker(text)

        chunks = []
        for idx, c in enumerate(chonks):
            chunk_text = getattr(c, "text", "").strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "token_count": getattr(c, "token_count", len(chunk_text)),
                    "chunk_index": idx,
                    "start_index": getattr(c, "start_index", 0),
                    "end_index": getattr(c, "end_index", len(chunk_text))
                })

        return chunks
    
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
    
    async def check_file_exists(self, filename: str) -> bool:
        """检查文件是否已经存在于 collection 中"""
        try:
            scroll_result = await self.qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename)
                    )]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            return len(scroll_result[0]) > 0
        except Exception as e:
            logger.error(f"检查文件存在性失败: {e}")
            return False
    
    async def ingest_file(self, file_path: Path):
        """导入单个文件"""
        filename = file_path.name
        
        # 检查文件是否已存在
        if await self.check_file_exists(filename):
            logger.info(f"⏭️  跳过已存在的文件: {filename}")
            return 0
        
        logger.info(f"📄 处理文件: {filename}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 分块
        chunks = self.chunk_text(content)
        logger.info(f"   分块数量: {len(chunks)}")
        
        # 处理每个分块
        points = []
        for idx, chunk in enumerate(chunks):
            # 提取文本内容
            chunk_text = chunk["text"]

            # 生成向量
            dense_vector = await self.embed_text(chunk_text)
            sparse_vector = self.build_sparse_vector(chunk_text)

            # 创建点
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "bm25": SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"]
                    )
                },
                payload={
                    "content": chunk_text,
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            )
            points.append(point)
        
        # 批量上传
        await self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"✅ 成功导入: {filename} ({len(points)} 个分块)")
        return len(points)
    
    async def ingest_all(self):
        """导入所有文件"""
        await self.ensure_collection_exists()

        md_files = list(DATA_DIR.glob("*.md"))
        logger.info(f"发现 {len(md_files)} 个 md 文件")

        total_chunks = 0
        for file_path in md_files:
            chunks_count = await self.ingest_file(file_path)
            total_chunks += chunks_count

        logger.info(f"\n🎉 导入完成！总共导入 {total_chunks} 个文本块")
    
    async def close(self):
        """关闭连接"""
        await self.qdrant_client.close()
        await self.openai_client.close()


async def main():
    """主函数"""
    ingestion = DataIngestion()
    try:
        await ingestion.ingest_all()
    finally:
        await ingestion.close()


if __name__ == "__main__":
    asyncio.run(main())
