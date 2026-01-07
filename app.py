"""
RAG Parameter Tuner - Flask Backend API
支持实时调节 system prompt、chunk size、overlap 和模型选择
"""
import os
import sys
import traceback
import logging

# 首先配置日志（必须在其他导入之前）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # 强制重新配置
)
logger = logging.getLogger(__name__)

# 输出启动信息
print("=" * 60, file=sys.stdout)
print("Starting RAG Parameter Tuner Application", file=sys.stdout)
print("=" * 60, file=sys.stdout)
logger.info("=" * 60)
logger.info("Starting app.py initialization...")
logger.info("=" * 60)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 尝试从 .env 文件加载环境变量
try:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
except Exception as e:
    logger.warning(f"无法加载 .env 文件: {e}")

# naive_rag.py 现在在同一目录中，直接导入
# 添加当前目录到路径（确保可以导入）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
logger.info(f"Current directory: {current_dir}")
logger.info(f"Python path: {sys.path[:3]}")

# 导入 RAG 相关模块
logger.info("Importing RAG modules...")
print("STEP: Starting RAG module import", flush=True)
sys.stdout.flush()

try:
    logger.info("  - Importing naive_rag...")
    print("STEP: Importing naive_rag module...", flush=True)
    sys.stdout.flush()
    
    from naive_rag import (
        get_rag_response, 
        create_system_prompt, 
        create_prompt_messages,
        answer_question_with_citations,
        init_vector_store,
        split_documents
    )
    logger.info("  - ✓ naive_rag imported")
    print("STEP: naive_rag imported successfully", flush=True)
    sys.stdout.flush()
    
    logger.info("  - Importing langchain modules...")
    print("STEP: Importing langchain modules...", flush=True)
    sys.stdout.flush()
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    logger.info("  - ✓ langchain modules imported")
    print("STEP: langchain modules imported successfully", flush=True)
    sys.stdout.flush()
except Exception as e:
    logger.error(f"Failed to import RAG modules: {e}")
    import traceback
    error_trace = traceback.format_exc()
    logger.error(error_trace)
    print(f"ERROR: Failed to import RAG modules: {e}", flush=True)
    print(error_trace, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    raise

# 导入 load_documents 函数
# load.py 现在在同一目录中，直接导入
# 当前目录已在路径中（之前已添加）
logger.info("Importing load_documents...")
try:
    from load import load_documents
    logger.info("  - ✓ load_documents imported")
except ImportError as e:
    logger.warning(f"load_documents not available: {e}")
    logger.warning("Some features that require load_documents may not work, but core RAG functionality should still work.")
    load_documents = None
except Exception as e:
    logger.error(f"Unexpected error importing load_documents: {e}")
    import traceback
    logger.error(traceback.format_exc())
    load_documents = None

# Flask 初始化：配置静态文件服务
# static_folder='.' 表示当前目录作为静态文件目录
# static_url_path='' 表示静态文件直接通过根路径访问
logger.info("Initializing Flask app...")
app = Flask(__name__, static_folder='.', static_url_path='')
logger.info("Flask app initialized successfully")
print("STEP: Flask app initialized", flush=True)
sys.stdout.flush()

# CORS 配置：允许跨域请求
# 生产环境应该限制为特定域名以提高安全性
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
if ALLOWED_ORIGINS == ['*']:
    # 开发环境：允许所有来源
    CORS(app)
else:
    # 生产环境：只允许指定的域名
    CORS(app, resources={
        r"/api/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })

# 全局变量存储向量库（避免重复加载）
_vector_store = None
_embeddings = None
_custom_vector_stores = {}  # 缓存不同chunk参数构建的向量库（OpenAIEmbedding）
_custom_retrieval_systems = {}  # 缓存不同chunk参数构建的检索系统（BM25, SPECTER, MedCPT, Contriever）

# 定义 naivetest_dir：指向当前目录（部署环境中索引可能不存在，会使用自定义构建）
naivetest_dir = os.path.dirname(os.path.abspath(__file__))

# 8篇核心文献的标准引用格式映射
FULL_CITATION_MAP = {
    'AAPMRCompendium_NP002': 'Cheng AL, Herman E, Abramoff B, et al. Multidisciplinary collaborative guidance on the assessment and treatment of patients with Long COVID: A compendium statement. PM&R. 2025;17(6):684-708. doi:10.1002/pmrj.13397',
    'Al-Aly_39122965': 'Al-Aly Z, Davis H, McCorkell L, et al. Long COVID science, research and policy. Nat Med. 2024;30(8):2148-2164. doi:10.1038/s41591-024-03173-6',
    'Bateman_34454716': 'Bateman L, Bested AC, Bonilla HF, et al. Myalgic Encephalomyelitis/Chronic Fatigue Syndrome: Essentials of Diagnosis and Management. Mayo Clinic Proceedings. 2021;96(11):2861-2878. doi:10.1016/j.mayocp.2021.07.004',
    'Fineberg_39110819': 'Committee on Examining the Working Definition for Long COVID, Board on Health Sciences Policy, Board on Global Health, Health and Medicine Division, National Academies of Sciences, Engineering, and Medicine. A Long COVID Definition: A Chronic, Systemic Disease State with Profound Consequences. (Fineberg HV, Brown L, Worku T, Goldowitz I, eds.). National Academies Press; 2024:27768. doi:10.17226/27768',
    'Mueller_40105889': 'Mueller MR, Ganesh R, Beckman TJ, Hurt RT. Long COVID: emerging pathophysiological mechanisms. Minerva Med. 2025;116(2). doi:10.23736/S0026-4806.25.09539-4',
    'Peluso_39326415': 'Peluso MJ, Deeks SG. Mechanisms of long COVID and the path toward therapeutics. Cell. 2024;187(20):5500-5529. doi:10.1016/j.cell.2024.07.054',
    'Vogel_39142505': 'Vogel JM, Pollack B, Spier E, et al. Designing and optimizing clinical trials for long COVID. Life Sciences. 2024;355:122970. doi:10.1016/j.lfs.2024.122970',
    'Zeraatkar_39603702': 'Zeraatkar D, Ling M, Kirsh S, et al. Interventions for the management of long covid (post-covid condition): living systematic review. BMJ. 2024;387:e081318. doi:10.1136/bmj-2024-081318'
}

def get_vector_store(chunk_size=1200, chunk_overlap=600):
    """获取或创建向量库（支持自定义chunk参数）"""
    global _vector_store, _embeddings, _custom_vector_stores
    
    # 确保参数是整数
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    
    # 如果是默认参数，尝试使用预构建的索引
    if chunk_size == 1200 and chunk_overlap == 600:
        if _vector_store is None:
            _embeddings = OpenAIEmbeddings()
            faiss_index_path = os.path.join(naivetest_dir, "faiss_index")
            if os.path.exists(faiss_index_path):
                logger.info(f"Loading pre-built FAISS index from {faiss_index_path}")
                _vector_store = FAISS.load_local(faiss_index_path, _embeddings, allow_dangerous_deserialization=True)
            else:
                # 如果预构建索引不存在，使用自定义构建方式（部署环境常见情况）
                logger.info(f"Pre-built FAISS index not found at {faiss_index_path}, building custom vector store...")
                _vector_store = build_custom_vector_store(chunk_size, chunk_overlap)
        return _vector_store
    else:
        # 对于自定义参数，需要重新加载和分割文档
        cache_key = f"{chunk_size}_{chunk_overlap}"
        if cache_key not in _custom_vector_stores:
            logger.info(f"Building custom vector store with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            try:
                _custom_vector_stores[cache_key] = build_custom_vector_store(chunk_size, chunk_overlap)
            except Exception as e:
                logger.error(f"Failed to build custom vector store: {e}")
                raise Exception(f"构建自定义向量库失败: {str(e)}")
        return _custom_vector_stores[cache_key]

def build_custom_vector_store(chunk_size, chunk_overlap):
    """使用自定义chunk参数重新加载文档并构建向量库"""
    try:
        # 确保参数是整数
        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)
        
        # 初始化embeddings
        global _embeddings
        if _embeddings is None:
            _embeddings = OpenAIEmbeddings()
        
        # 加载原始文档
        # 只加载longcovid_clinical corpus中的8篇核心文献，与预构建索引保持一致
        # 这8篇文献是：AAPMRCompendium_NP002, Al-Aly_39122965, Bateman_34454716,
        # Fineberg_39110819, Mueller_40105889, Peluso_39326415, Vogel_39142505, Zeraatkar_39603702
        logger.info("Loading original documents from corpus (longcovid_clinical subset)...")
        try:
            all_documents = load_documents(only_included=False)
            # 只保留longcovid_clinical corpus中的8篇核心文献
            longcovid_clinical_doc_ids = {
                'AAPMRCompendium_NP002',
                'Al-Aly_39122965',
                'Bateman_34454716',
                'Fineberg_39110819',
                'Mueller_40105889',
                'Peluso_39326415',
                'Vogel_39142505',
                'Zeraatkar_39603702'
            }
            documents = [doc for doc in all_documents if doc['id'] in longcovid_clinical_doc_ids]
            logger.info(f"Filtered to {len(documents)} documents from longcovid_clinical corpus")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise Exception(f"无法加载文档: {str(e)}")
        
        if not documents or len(documents) == 0:
            raise Exception("未找到任何文档，请检查corpus目录")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # 转换为 LangChain Document 对象
        langchain_docs = []
        for doc in documents:
            doc_id = doc['id']
            # 获取标准引用格式，如果不存在则使用short_citation
            full_citation = FULL_CITATION_MAP.get(doc_id, doc.get('short_citation', doc_id))
            langchain_docs.append(
                Document(
                    page_content=doc['content'],
                    metadata={
                        "id": doc_id,
                        "short_citation": doc['short_citation'],
                        "full_citation": full_citation,  # 添加标准引用格式
                        "categories": doc.get('categories', {})
                    }
                )
            )
        
        # 使用自定义参数分割文档
        logger.info(f"Splitting documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(langchain_docs)
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise Exception(f"文档分割失败: {str(e)}")
        
        if not chunks or len(chunks) == 0:
            raise Exception("文档分割后未生成任何chunks")
        
        # 为每个chunk添加正确的chunk ID（格式：文档ID_chunk索引）
        # RecursiveCharacterTextSplitter会自动继承父文档的metadata，但我们需要添加chunk索引
        # 按文档分组，为每个文档的chunks分配连续的索引（从1开始）
        doc_chunk_indices = {}  # 跟踪每个文档的下一个chunk索引
        for chunk in chunks:
            doc_id = chunk.metadata.get('id', 'unknown')
            if doc_id not in doc_chunk_indices:
                doc_chunk_indices[doc_id] = 1  # 第一个chunk索引为1
            else:
                doc_chunk_indices[doc_id] += 1  # 后续chunk索引递增
            
            # 更新chunk ID为：文档ID_chunk索引（从1开始，与原始格式一致）
            chunk_index = doc_chunk_indices[doc_id]
            chunk.metadata['id'] = f"{doc_id}_{chunk_index}"
        
        logger.info(f"Created {len(chunks)} chunks with proper chunk IDs")
        
        # 构建向量库（不保存到磁盘，只在内存中使用）
        logger.info("Building vector store...")
        try:
            vector_store = FAISS.from_documents(chunks, _embeddings)
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise Exception(f"构建向量库失败: {str(e)}")
        
        logger.info("Vector store built successfully")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error in build_custom_vector_store: {e}")
        logger.error(traceback.format_exc())
        raise

def build_custom_chunks(chunk_size, chunk_overlap):
    """使用自定义chunk参数重新加载文档并分割（用于非OpenAIEmbedding检索器）"""
    try:
        # 确保参数是整数
        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)
        
        # 加载原始文档
        logger.info("Loading original documents from corpus (longcovid_clinical subset)...")
        try:
            if load_documents is None:
                raise ImportError("load_documents is not available. Please ensure load.py is accessible.")
            all_documents = load_documents(only_included=False)
            # 只保留longcovid_clinical corpus中的8篇核心文献
            longcovid_clinical_doc_ids = {
                'AAPMRCompendium_NP002',
                'Al-Aly_39122965',
                'Bateman_34454716',
                'Fineberg_39110819',
                'Mueller_40105889',
                'Peluso_39326415',
                'Vogel_39142505',
                'Zeraatkar_39603702'
            }
            documents = [doc for doc in all_documents if doc['id'] in longcovid_clinical_doc_ids]
            logger.info(f"Filtered to {len(documents)} documents from longcovid_clinical corpus")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise Exception(f"无法加载文档: {str(e)}")
        
        if not documents or len(documents) == 0:
            raise Exception("未找到任何文档，请检查corpus目录")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # 转换为 LangChain Document 对象
        langchain_docs = []
        for doc in documents:
            doc_id = doc['id']
            # 获取标准引用格式，如果不存在则使用short_citation
            full_citation = FULL_CITATION_MAP.get(doc_id, doc.get('short_citation', doc_id))
            langchain_docs.append(
                Document(
                    page_content=doc['content'],
                    metadata={
                        "id": doc_id,
                        "short_citation": doc['short_citation'],
                        "full_citation": full_citation,
                        "categories": doc.get('categories', {})
                    }
                )
            )
        
        # 使用自定义参数分割文档
        logger.info(f"Splitting documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(langchain_docs)
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise Exception(f"文档分割失败: {str(e)}")
        
        if not chunks or len(chunks) == 0:
            raise Exception("文档分割后未生成任何chunks")
        
        # 为每个chunk添加正确的chunk ID（格式：文档ID_chunk索引）
        doc_chunk_indices = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get('id', 'unknown')
            if doc_id not in doc_chunk_indices:
                doc_chunk_indices[doc_id] = 1
            else:
                doc_chunk_indices[doc_id] += 1
            
            chunk_index = doc_chunk_indices[doc_id]
            chunk.metadata['id'] = f"{doc_id}_{chunk_index}"
        
        logger.info(f"Created {len(chunks)} chunks with proper chunk IDs")
        
        return chunks
    except Exception as e:
        logger.error(f"Error in build_custom_chunks: {e}")
        logger.error(traceback.format_exc())
        raise

def get_custom_retrieval_system(retriever_name, chunk_size, chunk_overlap):
    """获取或创建自定义chunk参数的检索系统（带缓存）"""
    global _custom_retrieval_systems
    
    # 确保参数是整数
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    
    # 创建缓存键
    cache_key = f"{retriever_name}_{chunk_size}_{chunk_overlap}"
    
    # 如果是默认参数，使用预构建的索引（通过iMedRAG的RetrievalSystem）
    if chunk_size == 1200 and chunk_overlap == 600:
        # 使用预构建索引，不需要缓存
        return None  # 返回None表示使用默认索引
    
    # 检查缓存
    if cache_key not in _custom_retrieval_systems:
        logger.info(f"Building custom {retriever_name} retrieval system with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        try:
            _custom_retrieval_systems[cache_key] = build_custom_retrieval_system(retriever_name, chunk_size, chunk_overlap)
        except Exception as e:
            logger.error(f"Failed to build custom retrieval system: {e}")
            raise Exception(f"构建自定义检索系统失败: {str(e)}")
    
    return _custom_retrieval_systems[cache_key]

def build_custom_retrieval_system(retriever_name, chunk_size, chunk_overlap):
    """为指定检索器构建自定义chunk参数的索引"""
    try:
        # 确保参数是整数
        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)
        
        # 构建chunks
        chunks = build_custom_chunks(chunk_size, chunk_overlap)
        
        # 根据检索器类型构建不同的索引
        if retriever_name == "BM25":
            # BM25: 构建倒排索引
            logger.info("Building BM25 index with custom chunk parameters...")
            import tempfile
            import json
            import subprocess
            
            # 创建临时目录存储索引
            temp_index_dir = tempfile.mkdtemp(prefix=f"bm25_custom_{chunk_size}_{chunk_overlap}_")
            temp_chunk_dir = os.path.join(temp_index_dir, "chunks")
            os.makedirs(temp_chunk_dir, exist_ok=True)
            
            # 将chunks转换为jsonl格式（按文档分组）
            doc_chunks = {}
            for chunk in chunks:
                doc_id = chunk.metadata.get('id', '').rsplit('_', 1)[0]  # 去掉chunk索引
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append({
                    "id": chunk.metadata.get('id', ''),
                    "title": chunk.metadata.get('short_citation', ''),
                    "content": chunk.page_content
                })
            
            # 写入jsonl文件
            for doc_id, chunk_list in doc_chunks.items():
                jsonl_path = os.path.join(temp_chunk_dir, f"{doc_id}.jsonl")
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for chunk_data in chunk_list:
                        f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
            
            # 构建Lucene索引
            index_path = os.path.join(temp_index_dir, "index")
            try:
                subprocess.run([
                    "python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", temp_chunk_dir,
                    "--index", index_path,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "4"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build BM25 index: {e.stderr.decode()}")
                raise Exception(f"构建BM25索引失败: {e.stderr.decode()}")
            
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher(index_path)
            
            # 创建自定义Retriever包装类
            class CustomBM25Retriever:
                def __init__(self, searcher, chunks, temp_dir):
                    self.searcher = searcher
                    self.chunks = chunks
                    self.temp_dir = temp_dir
                    self.chunk_map = {chunk.metadata.get('id', ''): chunk for chunk in chunks}
                
                def get_relevant_documents(self, question, k=25, id_only=False):
                    # 使用BM25搜索
                    hits = self.searcher.search(question, k=k)
                    texts = []
                    scores = []
                    for hit in hits:
                        doc_id = hit.docid
                        # 从chunk_map中获取chunk
                        chunk = self.chunk_map.get(doc_id)
                        if chunk:
                            texts.append({
                                "id": chunk.metadata.get('id', ''),
                                "title": chunk.metadata.get('short_citation', ''),
                                "content": chunk.page_content
                            })
                            scores.append(hit.score)
                    return texts, scores
            
            return CustomBM25Retriever(searcher, chunks, temp_index_dir)
            
        elif retriever_name in ["SPECTER", "Contriever", "MedCPT"]:
            # Dense检索器: 计算embeddings并构建FAISS索引
            logger.info(f"Building {retriever_name} index with custom chunk parameters...")
            
            # 导入iMedRAG的embed函数
            import sys
            imedrag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "src"))
            if imedrag_path not in sys.path:
                sys.path.insert(0, imedrag_path)
            
            from utils import CustomizeSentenceTransformer
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            import tempfile
            import json
            import torch
            
            # 确定模型名称
            model_map = {
                "SPECTER": "allenai/specter",
                "Contriever": "facebook/contriever",
                "MedCPT": "ncbi/MedCPT-Article-Encoder"
            }
            model_name = model_map[retriever_name]
            
            # 加载模型
            if "contriever" in model_name.lower():
                model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
            
            model.eval()
            
            # 准备文本和计算embeddings
            all_embeddings = []
            all_metadatas = []
            chunk_index = 0
            
            # 按文档分组处理
            doc_chunks = {}
            for chunk in chunks:
                doc_id = chunk.metadata.get('id', '').rsplit('_', 1)[0]
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk)
            
            with torch.no_grad():
                for doc_id, chunk_list in doc_chunks.items():
                    # 准备文本
                    if "specter" in model_name.lower():
                        texts = [model.tokenizer.sep_token.join([chunk.metadata.get('short_citation', ''), chunk.page_content]) for chunk in chunk_list]
                    elif "contriever" in model_name.lower():
                        texts = [". ".join([chunk.metadata.get('short_citation', ''), chunk.page_content]).replace('..', '.').replace("?.", "?") for chunk in chunk_list]
                    elif "medcpt" in model_name.lower():
                        # MedCPT使用双编码器，需要[[title, content]]格式
                        # CustomizeSentenceTransformer应该支持这种格式
                        texts = [[chunk.metadata.get('short_citation', ''), chunk.page_content] for chunk in chunk_list]
                    else:
                        from utils import concat
                        texts = [concat(chunk.metadata.get('short_citation', ''), chunk.page_content) for chunk in chunk_list]
                    
                    # 计算embeddings
                    try:
                        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                    except Exception as encode_error:
                        logger.error(f"Failed to encode texts for {retriever_name}: {encode_error}")
                        # 对于MedCPT，如果encode失败，尝试使用不同的格式
                        if "medcpt" in model_name.lower():
                            # MedCPT可能需要特殊处理，这里先抛出错误
                            raise Exception(f"MedCPT encoding failed: {encode_error}. MedCPT may require special handling.")
                        raise
                    all_embeddings.append(embeddings)
                    
                    # 保存metadata
                    for chunk in chunk_list:
                        all_metadatas.append({
                            "index": chunk_index,
                            "source": doc_id,
                            "id": chunk.metadata.get('id', '')
                        })
                        chunk_index += 1
            
            # 合并所有embeddings
            all_embeddings = np.vstack(all_embeddings)
            h_dim = all_embeddings.shape[1]
            
            # 构建FAISS索引
            index = faiss.IndexFlatIP(h_dim)  # Inner Product for cosine similarity
            # 归一化embeddings
            faiss.normalize_L2(all_embeddings)
            index.add(all_embeddings.astype('float32'))
            
            # 创建自定义Retriever包装类
            class CustomDenseRetriever:
                def __init__(self, index, metadatas, chunks, model, model_name):
                    self.index = index
                    self.metadatas = metadatas
                    self.chunks = chunks
                    self.model = model
                    self.model_name = model_name
                    self.chunk_map = {chunk.metadata.get('id', ''): chunk for chunk in chunks}
                
                def get_relevant_documents(self, question, k=25, id_only=False):
                    # 编码问题
                    if "specter" in self.model_name.lower():
                        query_text = self.model.tokenizer.sep_token.join([question, ""])
                    elif "contriever" in self.model_name.lower():
                        query_text = question
                    elif "medcpt" in self.model_name.lower():
                        # MedCPT使用双编码器，query应该是[question, ""]格式
                        query_text = [question, ""]
                    else:
                        query_text = question
                    
                    with torch.no_grad():
                        try:
                            query_embedding = self.model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)[0]
                        except Exception as encode_error:
                            logger.error(f"Failed to encode query for {self.model_name}: {encode_error}")
                            raise
                    
                    # 归一化
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    
                    # 搜索
                    scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
                    
                    texts = []
                    result_scores = []
                    for idx, score in zip(indices[0], scores[0]):
                        if idx < len(self.metadatas):
                            meta = self.metadatas[idx]
                            chunk_id = meta.get('id', '')
                            chunk = self.chunk_map.get(chunk_id)
                            if chunk:
                                texts.append({
                                    "id": chunk.metadata.get('id', ''),
                                    "title": chunk.metadata.get('short_citation', ''),
                                    "content": chunk.page_content
                                })
                                result_scores.append(float(score))
                    
                    return texts, result_scores
            
            return CustomDenseRetriever(index, all_metadatas, chunks, model, model_name)
        else:
            raise ValueError(f"Unsupported retriever: {retriever_name}")
            
    except Exception as e:
        logger.error(f"Error in build_custom_retrieval_system: {e}")
        logger.error(traceback.format_exc())
        raise

def get_rag_response_custom(
    question,
    system_prompt=None,
    chunk_size=1200,
    chunk_overlap=600,
    model="gpt-4o",
    retriever_name="OpenAIEmbedding",
    category=None,
    top_k=25,
    temperature=None
):
    """
    自定义参数的 RAG 响应生成
    
    Args:
        question: 用户问题
        system_prompt: 自定义 system prompt（如果为 None，使用默认）
        chunk_size: chunk 大小（对于OpenAIEmbedding检索器，会真正使用此参数重新分割文档并构建向量库）
        chunk_overlap: chunk 重叠大小（对于OpenAIEmbedding检索器，会真正使用此参数重新分割文档并构建向量库）
        model: 使用的模型（gpt-4o, gpt-5, gpt-5.2 等）
        retriever_name: 检索器名称
        category: 文档类别
        top_k: 检索的 chunk 数量
    
    Returns:
        dict: 包含 answer, chunks, metadata 的字典
    
    Note:
        - 对于OpenAIEmbedding检索器：会使用自定义chunk参数重新加载文档、分割并构建向量库
        - 对于其他检索器（BM25, Contriever等）：使用iMedRAG预建索引（默认chunk参数），chunk_size和chunk_overlap参数仅用于记录
    """
    try:
        # 对于非 OpenAIEmbedding 检索器，需要特殊处理
        if retriever_name != "OpenAIEmbedding":
            # 检查是否需要使用自定义chunk参数
            use_custom_chunks = (chunk_size != 1200 or chunk_overlap != 600)
            
            if use_custom_chunks:
                # 使用自定义chunk参数构建检索系统
                logger.info(f"Using custom chunk parameters (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}) for {retriever_name}")
                try:
                    custom_retriever = get_custom_retrieval_system(retriever_name, chunk_size, chunk_overlap)
                    if custom_retriever is None:
                        raise Exception("Failed to build custom retrieval system")
                    
                    # 使用自定义检索器检索
                    retrieved_snippets, scores = custom_retriever.get_relevant_documents(question, k=top_k, id_only=False)
                    
                    # 转换为 Document 格式
                    chunks = []
                    for snippet in retrieved_snippets:
                        # 从chunk中获取full_citation
                        chunk_id = snippet.get('id', '')
                        doc_id = chunk_id.rsplit('_', 1)[0] if '_' in chunk_id else chunk_id
                        full_citation = FULL_CITATION_MAP.get(doc_id, snippet.get('title', ''))
                        
                        doc = Document(
                            page_content=snippet.get('content', ''),
                            metadata={
                                'id': chunk_id,
                                'title': snippet.get('title', ''),
                                'short_citation': snippet.get('title', ''),
                                'full_citation': full_citation,
                                'categories': {}
                            }
                        )
                        chunks.append(doc)
                    
                    chunks_with_scores = list(zip(chunks, scores))
                    return get_rag_response_custom_direct(
                        question, chunks, system_prompt, model, retriever_name, category, 
                        chunk_size, chunk_overlap, top_k, chunks_with_scores, temperature
                    )
                except Exception as e:
                    logger.error(f"Custom retrieval system failed: {e}")
                    logger.warning(f"Falling back to default retrieval system for {retriever_name}")
                    # 如果自定义检索系统失败，回退到默认系统
                    use_custom_chunks = False
            
            if not use_custom_chunks:
                # 使用默认chunk参数（1200/600）的预构建索引
                # 导入 iMedRAG 相关模块
                import sys
                imedrag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "src"))
                if imedrag_path not in sys.path:
                    sys.path.insert(0, imedrag_path)
                
                try:
                    from utils import RetrievalSystem
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
                    retrieval_system = RetrievalSystem(
                        retriever_name=retriever_name,
                        corpus_name="longcovid_clinical",
                        db_dir=os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "corpus"),
                        cache=False,
                        HNSW=False
                    )
                    retrieved_snippets, scores = retrieval_system.retrieve(question, k=top_k, id_only=False)
                    
                    # 转换为 Document 格式
                    chunks = []
                    for snippet in retrieved_snippets:
                        chunk_id = snippet.get('id', '')
                        doc_id = chunk_id.rsplit('_', 1)[0] if '_' in chunk_id else chunk_id
                        full_citation = FULL_CITATION_MAP.get(doc_id, snippet.get('title', ''))
                        
                        doc = Document(
                            page_content=snippet.get('content', ''),
                            metadata={
                                'id': chunk_id,
                                'title': snippet.get('title', ''),
                                'short_citation': snippet.get('title', ''),
                                'full_citation': full_citation,
                                'categories': {}
                            }
                        )
                        chunks.append(doc)
                    
                    chunks_with_scores = list(zip(chunks, scores))
                    return get_rag_response_custom_direct(
                        question, chunks, system_prompt, model, retriever_name, category, 
                        chunk_size, chunk_overlap, top_k, chunks_with_scores, temperature
                    )
                except Exception as e:
                    # 如果 iMedRAG 检索失败，回退到使用原始函数
                    logger.warning(f"iMedRAG retrieval failed: {e}. Falling back to default method.")
                    result = get_rag_response(
                        question=question,
                        verbose=False,
                        history=None,
                        category=category,
                        retriever_name=retriever_name,
                        corpus_name="longcovid_clinical"
                    )
                    
                    # 提取 chunks（如果可用）
                    chunks = []
                    if hasattr(result, 'chunks'):
                        chunks = result.chunks
                    elif isinstance(result, dict) and 'chunks' in result:
                        chunks = result['chunks']
                    
                    if chunks:
                        return get_rag_response_custom_direct(
                            question, chunks, system_prompt, model, retriever_name, category, 
                            chunk_size, chunk_overlap, top_k, None, temperature
                        )
                    else:
                        # 如果无法获取 chunks，直接返回结果
                        return {
                            "answer": result.get('answer', '') if isinstance(result, dict) else str(result),
                            "chunks": [],
                            "metadata": {
                                "model": model,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "num_chunks": 0,
                                "retriever_name": retriever_name,
                                "category": category,
                                "temperature": temperature if temperature is not None else (1.0 if model == 'gpt-5' else 0.2)
                            }
                        }
        
        # OpenAIEmbedding 检索器：使用自定义参数重新构建向量库（如果需要）
        try:
            vector_store = get_vector_store(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as e:
            logger.error(f"Failed to get vector store: {e}")
            return {
                "error": f"无法构建向量库: {str(e)}",
                "traceback": traceback.format_exc()
            }
        
        # 检索相关文档
        try:
            chunks_with_scores = vector_store.similarity_search_with_score(question, k=top_k)
            chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[1])
            chunks = [chunk for chunk, _ in chunks_with_scores[:top_k]]
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return {
                "error": f"检索失败: {str(e)}",
                "traceback": traceback.format_exc()
            }
        
        return get_rag_response_custom_direct(
            question, chunks, system_prompt, model, retriever_name, category, chunk_size, chunk_overlap, top_k, chunks_with_scores, temperature
        )
    
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def get_rag_response_custom_direct(
    question, chunks, system_prompt, model, retriever_name, category, 
    chunk_size, chunk_overlap, top_k, chunks_with_scores=None, temperature=None
):
    """直接生成答案的辅助函数"""
    # 创建 prompt
    if system_prompt is None:
        # 使用默认 system prompt
        available_citations = list(set([chunk.metadata.get('id', '') for chunk in chunks if hasattr(chunk, 'metadata')]))
        system_prompt = create_system_prompt(
            category=category,
            retriever_name=retriever_name,
            corpus_name="longcovid_clinical",
            available_citations=available_citations
        )
    
    # 创建用户消息
    user_content_parts = [f"Question: {question}"]
    
    # 添加文档内容
    if chunks:
        document_texts = "\n\n".join([
            f"---Document ({chunk.metadata['id'] if hasattr(chunk, 'metadata') else 'unknown'})\n"
            f"---short_citation: {chunk.metadata.get('short_citation', chunk.metadata.get('title', '')) if hasattr(chunk, 'metadata') else ''}\n"
            f"{chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)}"
            for chunk in chunks
        ])
        user_content_parts.append(document_texts)
    
    user_content_parts.append("Answer:")
    user_content = "\n\n".join(user_content_parts)
    
    # 创建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # 调用 LLM
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    # Determine temperature to use
    # GPT-5 only supports default temperature (1), so ignore user-specified temperature for gpt-5
    # For other models, use user-specified temperature or default to 0.2
    if model == 'gpt-5':
        # GPT-5 only supports default temperature, don't set it explicitly
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
    else:
        # Use user-specified temperature, or default to 0.2 if not specified
        actual_temperature = temperature if temperature is not None else 0.2
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=actual_temperature
        )
    
    answer = response.choices[0].message.content
    
    # 处理 chunks_with_scores
    if chunks_with_scores is None:
        chunks_with_scores = [(chunk, 0.0) for chunk in chunks]
    
    # 返回结果
    chunks_data = []
    for chunk, score in chunks_with_scores[:top_k]:
        chunk_id = chunk.metadata.get('id', '') if hasattr(chunk, 'metadata') else ''
        # 从chunk ID中提取文档ID（去掉chunk索引部分，如"AAPMRCompendium_NP002_1" -> "AAPMRCompendium_NP002"）
        doc_id = chunk_id.rsplit('_', 1)[0] if '_' in chunk_id else chunk_id
        # 获取标准引用格式
        full_citation = chunk.metadata.get('full_citation', '') if hasattr(chunk, 'metadata') else ''
        if not full_citation:
            # 如果metadata中没有full_citation，从映射中获取
            full_citation = FULL_CITATION_MAP.get(doc_id, chunk.metadata.get('short_citation', chunk_id) if hasattr(chunk, 'metadata') else chunk_id)
        
        chunks_data.append({
            "id": chunk_id,
            "doc_id": doc_id,  # 添加文档ID用于映射
            "full_citation": full_citation,  # 添加标准引用格式
            "content": (chunk.page_content if hasattr(chunk, 'page_content') else str(chunk))[:200] + "...",
            "score": float(score)
        })
    
    return {
        "answer": answer,
        "chunks": chunks_data,
        "metadata": {
            "model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks),
            "retriever_name": retriever_name,
            "category": category,
            "temperature": temperature if temperature is not None else (1.0 if model == 'gpt-5' else 0.2)
        }
    }

@app.route('/api/generate', methods=['POST'])
def generate_answer():
    """生成答案的 API 端点"""
    try:
        data = request.json
        
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        
        # 获取参数（使用默认值）
        system_prompt = data.get('system_prompt', None)
        # 确保chunk_size和chunk_overlap是整数
        try:
            chunk_size = int(data.get('chunk_size', 1200))
            chunk_overlap = int(data.get('chunk_overlap', 600))
            top_k = int(data.get('top_k', 25))
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"参数类型错误: chunk_size, chunk_overlap 和 top_k 必须是整数. {str(e)}"}), 400
        
        model = data.get('model', 'gpt-4o')
        retriever_name = data.get('retriever_name', 'OpenAIEmbedding')
        category = data.get('category', None)
        
        # 获取temperature参数
        try:
            temperature = data.get('temperature', None)
            if temperature is not None:
                temperature = float(temperature)
                if temperature < 0 or temperature > 2:
                    return jsonify({"error": "temperature必须在0-2之间"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "temperature必须是数字"}), 400
        
        # 验证模型名称
        if model not in ['gpt-5', 'gpt-5.2', 'gpt-4o', 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']:
            return jsonify({"error": f"不支持的模型: {model}"}), 400
        
        # 生成答案
        result = get_rag_response_custom(
            question=question,
            system_prompt=system_prompt,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=model,
            retriever_name=retriever_name,
            category=category,
            top_k=top_k,
            temperature=temperature
        )
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/defaults', methods=['GET'])
def get_defaults():
    """获取默认参数"""
    # 获取默认的system prompt（不包含available_citations，因为初始化时还没有chunks）
    try:
        default_system_prompt = create_system_prompt(
            category=None,
            retriever_name="OpenAIEmbedding",
            corpus_name="longcovid_clinical",
            available_citations=None  # 不提供citations，返回基础的prompt
        )
    except Exception as e:
        logger.warning(f"Failed to generate default system prompt: {e}")
        default_system_prompt = ""
    
    return jsonify({
        "chunk_size": 1200,
        "chunk_overlap": 600,
        "model": "gpt-4o",
        "retriever_name": "OpenAIEmbedding",
        "top_k": 25,
        "temperature": 0.2,
        "system_prompt": default_system_prompt
    })

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "message": "Service is running"})

@app.route('/')
def index():
    """服务前端页面"""
    response = send_from_directory('.', 'index.html')
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@app.route('/<path:path>')
def serve_static(path):
    """服务静态文件（CSS、JS、图片等）"""
    # 排除 API 路由，这些应该由上面的路由处理
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    
    # 尝试发送静态文件
    try:
        return send_from_directory('.', path)
    except Exception as e:
        # 如果文件不存在，返回 index.html（用于前端路由）
        # 这样前端可以使用 HTML5 History API 进行路由
        if path not in ['favicon.ico']:
            return send_from_directory('.', 'index.html')
        raise

if __name__ == '__main__':
    # 检查 API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("警告: OPENAI_API_KEY 环境变量未设置")
    
    # 支持生产环境部署（从环境变量读取端口，如 Render、Railway 等）
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

