import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import yaml
import json
from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 配置类 ----
@dataclass
class RAGConfig:
    """RAG 管道配置参数"""
    # 嵌入模型配置
    embedding_model: str = "text2vec-base-chinese"
    # embedding_model: str ="all-MiniLM-L6-v2"
    # 检索配置
    search_top_k: int = 3
    search_type: str = "similarity"
    
    # 模型服务配置
    llm_model_name: str = "Qwen/QwQ-32B"
    llm_api_base: str = "https://api-inference.modelscope.cn/v1/"
    llm_streaming: bool = True
    llm_temperature: float = 0.3
    
    # 提示模板
    prompt_template: str = """基于以下上下文信息，请以专业的方式回答问题。如果无法从上下文中获得答案，请明确说明。
    
    上下文：
    {context}
    
    问题：{question}
    
    请用中文提供结构清晰的回答："""
    
    # 文档处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # 缓存配置
    cache_file_path: Path = field(default_factory=lambda: Path("vector_store_cache.faiss"))
    
    # 安全配置
    allow_dangerous_deserialization: bool = True
    
    # 调试配置
    verbose: bool = True

    def __post_init__(self) -> None:
        """参数验证"""
        if self.search_top_k <= 0:
            raise ValueError("search_top_k 必须为正整数")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 不能大于 chunk_size")
        if not (0 <= self.llm_temperature <= 2):
            raise ValueError("temperature 必须在 0-2 之间")

    @classmethod
    def from_yaml(cls, file_path: Path) -> "RAGConfig":
        """从 YAML 文件加载配置"""
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # 确保 cache_file_path 是 Path 对象
            if isinstance(config_data.get('cache_file_path'), str):
                config_data['cache_file_path'] = Path(config_data['cache_file_path'])
            
            return cls(**config_data)
        except Exception as e:
            logger.error(f"YAML 配置加载失败: {e}")
            raise

    @classmethod
    def from_json(cls, file_path: Path) -> "RAGConfig":
        """从 JSON 文件加载配置"""
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            # 确保 cache_file_path 是 Path 对象
            if isinstance(config_data.get('cache_file_path'), str):
                config_data['cache_file_path'] = Path(config_data['cache_file_path'])
            
            return cls(**config_data)
        except Exception as e:
            logger.error(f"JSON 配置加载失败: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于持久化）"""
        return self.__dict__

# ---- 嵌入模型封装 ----
class SentenceTransformerEmbeddings(Embeddings):
    """自定义句子嵌入模型封装"""
    def __init__(self, config: RAGConfig) -> None:
        self.model = SentenceTransformer(config.embedding_model)
        logger.info(f"初始化嵌入模型: {config.embedding_model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

# ---- 模块初始化函数 ----
def initialize_embeddings(config: RAGConfig) -> SentenceTransformerEmbeddings:
    """初始化嵌入模型"""
    return SentenceTransformerEmbeddings(config)

def initialize_vector_store(
    texts: List[str], 
    embeddings: Embeddings,
    config: RAGConfig
) -> FAISS:
    """初始化向量数据库"""
    try:
        # 检查缓存文件是否存在
        if config.cache_file_path.exists():
            logger.warning("注意：正在启用不安全的反序列化选项。确保你信任缓存文件的来源！")
            vector_store = FAISS.load_local(
                str(config.cache_file_path.parent), 
                embeddings, 
                index_name=config.cache_file_path.stem, 
                allow_dangerous_deserialization=config.allow_dangerous_deserialization
            )
        else:
            logger.info("正在创建向量存储...")
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=[{"source": f"doc_{i}"} for i in range(len(texts))]
            )
            # 保存到缓存文件
            vector_store.save_local(str(config.cache_file_path.parent), index_name=config.cache_file_path.stem)
            logger.info(f"向量存储已保存到缓存文件: {config.cache_file_path}")
        return vector_store
    except Exception as e:
        logger.error(f"向量存储初始化失败: {e}")
        raise

def initialize_llm(config: RAGConfig) -> ChatOpenAI:
    """初始化大语言模型"""
    # api_key = os.getenv("OPENAI_API_KEY", "XXXXXXX")
    api_key = os.getenv("OPENAI_API_KEY")
    
    # api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未找到 OPENAI_API_KEY 环境变量")

    return ChatOpenAI(
        openai_api_base=config.llm_api_base,
        openai_api_key=api_key,
        model_name=config.llm_model_name,
        streaming=config.llm_streaming,
        temperature=config.llm_temperature
    )

def build_qa_chain(
    vector_store: FAISS,
    llm: ChatOpenAI,
    config: RAGConfig
) -> RetrievalQA:
    """构建检索问答链"""
    # 配置检索器
    retriever = vector_store.as_retriever(
        search_type=config.search_type,
        search_kwargs={"k": config.search_top_k}
    )
    
    # 构建提示模板
    prompt = PromptTemplate(
        template=config.prompt_template,
        input_variables=["context", "question"]
    )
    
    # 配置问答链
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": config.verbose
        }
    )
import pandas as pd

def read_excel_qa_pairs(file_paths: List[str]) -> List[str]:
    corpus = []
    seen = set()
    for file_path in file_paths:
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        for _, row in df.iterrows():
            text = f"问题：{row['问题']}\n答案：{row['答案']}"
            if text not in seen:
                corpus.append(text)
                seen.add(text)
    # print(len(corpus))
    return corpus
    
def main(config: RAGConfig) -> None:
    """RAG 主流程"""
    # 读取Excel文件
    excel_files = ["全网运行指标问题.xlsx", "数字人问答.xlsx"]
    corpus = read_excel_qa_pairs(excel_files)
    
    try:
        # 初始化各组件
        embeddings = initialize_embeddings(config)
        vector_store = initialize_vector_store(corpus, embeddings, config)
        llm = initialize_llm(config)
        qa_chain = build_qa_chain(vector_store, llm, config)
        
        # 执行查询
        query = "数据中心是什么？"
        logger.info(f"正在处理查询: {query}")
        
        # 获取完整响应
        response = qa_chain({"query": query})
        
        # 输出结果
        print("\n=== 最终回答 ===")
        print(response["result"])
        
        # 显示参考来源
        if config.verbose:
            print("\n=== 参考文档 ===")
            for doc in response["source_documents"]:
                print(f"- {doc.page_content[:80]}...")

    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        raise

if __name__ == "__main__":
    # 使用方式 1: 默认配置
    main(RAGConfig())
    
    # # 使用方式 2: 从 YAML 加载配置
    # config = RAGConfig.from_yaml(Path("config.yml"))
    # main(config)
    
    # # 使用方式 3: 从 JSON 加载配置
    # config = RAGConfig.from_json(Path("config.json"))
    # main(config)
