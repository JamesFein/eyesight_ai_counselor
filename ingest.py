"""
数据导入脚本
将 data 目录中的 txt 文件导入到 Qdrant 的 eye_ana_guide collection
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, Modifier, SparseVector
from openai import AsyncOpenAI
from chonkie import SentenceChunker
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
DATA_DIR = Path(r"C:\Users\Administrator\Desktop\fast-gzmdrw-chat\data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "520"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_DELIMITERS = ["。", "！", "？", ". ", "! ", "? ", "\n"]
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
        
        # 初始化 Chonkie 分块器
        self.chunker = SentenceChunker(
            tokenizer_or_token_counter="character",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            delim=CHUNK_DELIMITERS,
            include_delim="prev"
        )
        
        logger.info("数据导入服务初始化完成")
    
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
    
    def chunk_text(self, text: str) -> List[str]:
        """使用 Chonkie 分块文本"""
        chonks = self.chunker(text)
        chunks = [getattr(c, "text", "").strip() for c in chonks if getattr(c, "text", "").strip()]
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
            # 生成向量
            dense_vector = await self.embed_text(chunk)
            sparse_vector = self.build_sparse_vector(chunk)
            
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
                    "content": chunk,
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
        
        txt_files = list(DATA_DIR.glob("*.txt"))
        logger.info(f"发现 {len(txt_files)} 个 txt 文件")
        
        total_chunks = 0
        for file_path in txt_files:
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
