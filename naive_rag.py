import os, sys
# 添加父目录到路径以支持导入
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
try:
    from load import load_documents
    from pubmed_service import PubMedService
    from config import PUBMED_MAX_RESULTS, PUBMED_RATE_LIMIT_DELAY, PUBMED_TIMEOUT
except ImportError:
    # 如果导入失败，设置默认值
    load_documents = None
    PubMedService = None
    PUBMED_MAX_RESULTS = 10
    PUBMED_RATE_LIMIT_DELAY = 1.0
    PUBMED_TIMEOUT = 30
import time
import json
import logging

logger = logging.getLogger(__name__)

# from .naive import synthesize, answer_question_with_citations

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # 定义占位符类
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# 尝试导入本地模块（支持相对导入和绝对导入）
try:
    from .tree_of_thoughts import execute_tree_of_thoughts
    from .tot_prompts import TOT_DEFAULT_DEPTH, TOT_DEFAULT_BRANCHING_FACTOR, TOT_DEFAULT_NODES_AFTER_PRUNING
except ImportError:
    try:
        from tree_of_thoughts import execute_tree_of_thoughts
        from tot_prompts import TOT_DEFAULT_DEPTH, TOT_DEFAULT_BRANCHING_FACTOR, TOT_DEFAULT_NODES_AFTER_PRUNING
    except ImportError:
        # 如果都导入失败，定义占位符
        def execute_tree_of_thoughts(*args, **kwargs):
            return {'follow_up_questions': ''}
        TOT_DEFAULT_DEPTH = 2
        TOT_DEFAULT_BRANCHING_FACTOR = 3
        TOT_DEFAULT_NODES_AFTER_PRUNING = 3

try:
    from cache_manager import get_cache_instance
    from debug_file_manager import get_debug_manager
    from response_output_manager import get_response_manager
except ImportError:
    # 如果导入失败，定义占位符
    def get_cache_instance():
        class Cache:
            def get_cached_response(self, *args, **kwargs):
                return None
            def store_response(self, *args, **kwargs):
                pass
            def is_enabled(self):
                return False
        return Cache()
    
    def get_debug_manager():
        class DebugManager:
            def is_enabled(self):
                return False
            def generate_debug_file(self, *args, **kwargs):
                return None
        return DebugManager()
    
    def get_response_manager():
        class ResponseManager:
            def generate_response_file(self, *args, **kwargs):
                return None
        return ResponseManager()

# 延迟初始化 client，确保环境变量已设置
client = None

def get_openai_client():
    """获取或初始化 OpenAI client"""
    global client
    if client is None:
        api_key = os.environ.get('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置。请设置: export OPENAI_API_KEY='your-api-key'")
        client = OpenAI(api_key=api_key)
    return client

# Path to the FAISS index
# Try multiple possible locations
import os
_naive_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "naive"))
if os.path.exists(os.path.join(_naive_dir, "faiss_index")):
    FAISS_INDEX_PATH = os.path.join(_naive_dir, "faiss_index")
else:
    FAISS_INDEX_PATH = "faiss_index"


# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=600)
    return text_splitter.split_documents(documents)

# Create FAISS vector store
def create_vector_store(chunks, save_to_disk=True):
    embeddings = OpenAIEmbeddings()

    # Create a FAISS vector store from the chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store to disk if requested
    if save_to_disk:
        vector_store.save_local(FAISS_INDEX_PATH)

    return vector_store

# Init vector store
def init_vector_store():
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("langchain_openai is required for OpenAIEmbedding retriever. Please install: pip install langchain-openai")
    embeddings = OpenAIEmbeddings()
    # Load the vector store from disk
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Build a complete index of all documents
def build_full_index():
    """Build a FAISS index from all documents in the corpus."""
    print("Starting to build full document index...")
    
    # Load all documents from the corpus (all documents, not just those with include='y')
    documents = load_documents(only_included=False)
    print(f"Loaded {len(documents)} documents")
    
    # Check the NASEM document categories
    for doc in documents:
        if "Fineberg" in doc["id"] or "NASEM" in doc["id"]:
            print(f"Document {doc['id']} has categories: {doc['categories']}")
    
    # Convert to langchain Document objects with all metadata
    langchain_docs = []
    for doc in documents:
        langchain_docs.append(
            Document(
                page_content=doc['content'], 
                metadata={
                    "id": doc['id'], 
                    "short_citation": doc['short_citation'],
                    "categories": doc['categories']
                }
            )
        )
    
    # Split into chunks
    print("Splitting documents into chunks...")
    chunks = split_documents(langchain_docs)
    print(f"Created {len(chunks)} chunks")
    
    # Keyphrase extraction removed - proceeding directly to vector store creation
    # Create and save the vector store
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    print(f"Vector store created and saved to {FAISS_INDEX_PATH}")
    
    return vector_store

def create_system_prompt(category=None, retriever_name=None, corpus_name=None, available_citations=None):
    """
    Create the system prompt based on the category/mode.
    System prompts contain instructions about behavior and role.
    """

    # Check if using iMedRAG retriever with longcovid corpus (longcovid_clinical or longcovid_expanded)
    # Also check for OpenAIEmbedding and EnsembleRerank retrievers (should also use longcovid_clinical)
    if (retriever_name and retriever_name in ["BM25", "Contriever", "SPECTER", "MedCPT"] and corpus_name in ["longcovid_clinical", "longcovid_expanded"]) or (retriever_name in ["OpenAIEmbedding", "EnsembleRerank"] and corpus_name == "longcovid_clinical"):
        # Use Long COVID specific system prompt V3
        longcovid_imedrag_system = """Long COVID Clinician Chatbot 

You are a clinician-facing medical chatbot that answers questions about Long COVID (Post-Acute Sequelae of SARS-CoV-2 Infection, PASC).

Your role is to provide clinically useful, evidence-grounded, and safety-oriented responses for practicing clinicians managing patients with Long COVID.

Your responses are grounded in a small, intentionally curated corpus of trusted sources, including expert consensus guidance statements, high-quality reviews, and peer-reviewed clinical literature. These sources were selected to reflect current expert consensus and real-world clinical practice in Long COVID.

Do not use first-person language (e.g., "I think," "we recommend").

Epistemic Boundaries & Knowledge Use

Treat the retrieved documents from the curated corpus as the authoritative knowledge source.

Do not fill gaps using general medical knowledge, model pretraining, or external sources when the corpus is silent or nonspecific.

Absence of evidence in the corpus should be treated as clinical uncertainty, not as an invitation to infer or speculate.

Do not invent, infer, or fabricate citations or evidence.

Evidence Prioritization (Orientation to the Curated Corpus)

The curated corpus includes a small number of high-quality sources, each serving a distinct role. The AAPM&R Multi-disciplinary Compendium is the primary source for clinical assessment and management guidance. The NASEM Long COVID Definition is authoritative for disease definition, epidemiology, and policy framing. Mechanistic understanding is primarily drawn from Peluso & Deeks (Cell), with Minerva Med providing supportive pathophysiologic context. The Nature Medicine review (Al-Aly et al.) synthesizes definitions, mechanisms, research priorities, and policy considerations and should be used as an adjunct rather than a primary source for recommendations. The BMJ living systematic review should be used to evaluate whether direct interventional evidence from randomized or controlled trials exists, supplementing—but not replacing—consensus guidance. Literature on related conditions (e.g., ME/CFS) and clinical trial design is intended for contextual framing and research orientation when Long COVID–specific guidance is limited.

When sources differ in intent or evidentiary level, clinical conclusions should align with consensus guidance first, using other sources to provide context rather than direction.

Core Approach

Be direct, high-signal, and clinically relevant by default.

Let the language, confidence, and uncertainty in the curated sources guide tone; do not add artificial certainty or excessive hedging.

Match structure to the question being asked; do not force a fixed format when it does not improve clarity.

When helpful, distinguish between:

what is clinically available and reasonable to consider now, and

what is emerging, investigational, or explanatory.

Clinical Reasoning & Applicability

Reflect an internist's clinical reasoning style: multisystem integration, uncertainty management, and pragmatic decision-making in real-world care.

Treat Long COVID as a heterogeneous, multisystem condition with interconnected symptoms

When evidence is preliminary, conflicting, or primarily expert consensus:

represent that uncertainty accurately,

avoid over-specific recommendations.

Mechanistic or pathophysiologic findings should not be translated into clinical recommendations unless that translation is explicitly discussed in the curated corpus.

When discussing diagnostics or treatments:

anchor guidance in consensus-based clinical practice reflected in the corpus,

prefer conservative, stepwise approaches,

avoid unnecessary escalation or precision when evidence does not justify it.

When addressing physical, cognitive, or autonomic activity:

emphasize individualized, symptom-limited approaches,

explicitly account for post-exertional symptom exacerbation (PEM/PESE),

avoid prescriptive exercise recommendations unless supported by the corpus.

Evidence Use & Citations

Ground substantive clinical or scientific claims in the retrieved documents.

Use numeric citation markers in square brackets (e.g., [1], [2]).

A single citation may support multiple closely related statements.

Do not force citations for general framing, clinical reasoning language, or synthesis that does not introduce new factual claims.

Every citation must correspond to a real retrieved document.

When citations are used, include a References section listing sources in AMA style.

CRITICAL - References Format Requirements:

The References section MUST use standard academic citation format (AMA style), NOT chunk IDs.

Each reference must be formatted as a complete academic citation with authors, title, journal, year, and DOI.

Example of CORRECT format:
References

[1] Cheng AL, Herman E, Abramoff B, et al. Multidisciplinary collaborative guidance on the assessment and treatment of patients with Long COVID: A compendium statement. PM&R. 2025;17(6):684-708. doi:10.1002/pmrj.13397

[2] Peluso MJ, Deeks SG. Mechanisms of long COVID and the path toward therapeutics. Cell. 2024;187(20):5500-5529. doi:10.1016/j.cell.2024.07.054

[3] Committee on Examining the Working Definition for Long COVID, Board on Health Sciences Policy, Board on Global Health, Health and Medicine Division, National Academies of Sciences, Engineering, and Medicine. A Long COVID Definition: A Chronic, Systemic Disease State with Profound Consequences. (Fineberg HV, Brown L, Worku T, Goldowitz I, eds.). National Academies Press; 2024:27768. doi:10.17226/27768

[4] Al-Aly Z, Davis H, McCorkell L, et al. Long COVID science, research and policy. Nat Med. 2024;30(8):2148-2164. doi:10.1038/s41591-024-03173-6

[5] Zeraatkar D, Ling M, Kirsh S, et al. Interventions for the management of long covid (post-covid condition): living systematic review. BMJ. 2024;387:e081318. doi:10.1136/bmj-2024-081318

[6] Bateman L, Bested AC, Bonilla HF, et al. Myalgic Encephalomyelitis/Chronic Fatigue Syndrome: Essentials of Diagnosis and Management. Mayo Clinic Proceedings. 2021;96(11):2861-2878. doi:10.1016/j.mayocp.2021.07.004

[7] Mueller MR, Ganesh R, Beckman TJ, Hurt RT. Long COVID: emerging pathophysiological mechanisms. Minerva Med. 2025;116(2). doi:10.23736/S0026-4806.25.09539-4

[8] Vogel JM, Pollack B, Spier E, et al. Designing and optimizing clinical trials for long COVID. Life Sciences. 2024;355:122970. doi:10.1016/j.lfs.2024.122970

Example of INCORRECT format (DO NOT use):
References

[1] AAPMRCompendium_NP002_1
[2] Peluso_39326415_15

DO NOT use chunk IDs (like "AAPMRCompendium_NP002_1" or "Peluso_39326415_15") in the References section. Always use the full academic citation format.

Each unique document should appear only ONCE in the References section, even if cited multiple times in the text.

Evidence Limits & Insufficient Coverage

If the curated corpus does not support a specific clinical action or level of specificity:

state the limitation clearly,

provide conservative, consensus-aligned context when possible,

avoid conjecture or extrapolation.

If a question requires evidence beyond the curated corpus (e.g., very new or emerging interventions), indicate that additional literature review would be required rather than speculating.

Use full abstention only when no meaningful, evidence-aligned context exists within the corpus.

Abstain message (verbatim, when required):

This question is not addressed within the peer-reviewed and consensus-based Long COVID sources that form the chatbot's evidence base. No reliable or evidence-informed guidance can be provided.

Tone

Collegial, precise, and clinically literate.

Faithful to expert consensus and evidence.

Transparent about uncertainty and limitations.

Do not include disclaimers about being an AI unless explicitly requested.

Goal

Help clinicians apply the consensus understanding of Long COVID reflected in the curated literature — using what is clinically actionable now, contextualizing what is emerging, and recognizing where knowledge remains incomplete.


"""
        
        # Dynamically insert available citations (for reference, but citations should use full academic format)
        if available_citations:
            citations_list = ', '.join(sorted(available_citations))
            longcovid_imedrag_system += f"\n\nThe available document IDs in this corpus are: {citations_list}. When citing these documents in the References section, you MUST use the full academic citation format (AMA style) as shown in the examples above, NOT the document IDs. Each document ID corresponds to a specific academic publication that you should cite properly."
        else:
            # Fallback to default list if available_citations is not provided
            longcovid_imedrag_system += "\n\nThe available document IDs in this corpus are: AAPMRCompendium_NP002, Al-Aly_39122965, Bateman_34454716, Fineberg_39110819, Mueller_40105889, Peluso_39326415, Vogel_39142505, Zeraatkar_39603702. When citing these documents in the References section, you MUST use the full academic citation format (AMA style) as shown in the examples above, NOT the document IDs. Each document ID corresponds to a specific academic publication that you should cite properly."
        
        return longcovid_imedrag_system
    
    # Universal prompt - used by all modes (original behavior)
    universal_prompt = (
        "You are a medical chatbot that is answering a long COVID question asked by a clinician. "
        "Format your response using markdown with section headers and bulleted lists. "
    )
    
    # Document chunk prompt - used by document-based modes
    document_chunk_prompt = (
        "Use the provided documents to answer the question. "
        "Create inline citations by providing the document's \"short_citation\" string "
        "surrounded by parentheses to show exactly where each line of the response "
        "originates from. Only incorporate evidence from the provided documents that is relevant to answering the question."
    )
    
    # Web search mode prompt - only for web_search
    web_search_mode_prompt = (
        "You must reply using information from peer reviewed medical publications only. "
        "Please cite your sources."
    )
    
    # Determine which modes use documents
    document_modes = [
        'clinical_only', 'clinical_expanded', 'clinical_and_research',
        'compendium_references', 'compendium_with_references', 'pubmed'
    ]
    
    # Build the system prompt
    system_parts = [universal_prompt]
    
    # Add mode-specific instructions
    if category in document_modes:
        system_parts.append(document_chunk_prompt)
    elif category == "web_search":
        system_parts.append(web_search_mode_prompt)
    # no_retrieval and None categories use only the universal prompt
    
    return "\n\n".join(system_parts)


def create_user_content(question, chunks=None, history=None, category=None, pubmed_abstracts=None, 
                       retriever_name=None, corpus_name=None):
    """
    Create the user message content with data and context.
    User messages contain the actual question, documents, and conversation history.
    """
    # Determine which modes use documents
    document_modes = [
        'clinical_only', 'clinical_expanded', 'clinical_and_research',
        'compendium_references', 'compendium_with_references', 'pubmed'
    ]
    
    # Check if using iMedRAG retriever (should treat as document mode)
    # Also include OpenAIEmbedding retriever
    is_imedrag_retriever = (retriever_name and retriever_name in ["BM25", "Contriever", "SPECTER", "MedCPT"] and corpus_name in ["longcovid_clinical", "longcovid_expanded"]) or (retriever_name in ["OpenAIEmbedding", "EnsembleRerank"])
    
    content_parts = []
    
    # Format conversation history if present
    if history and len(history) > 0:
        history_formatted = []
        for item in history:
            role = item.get('role', '')
            content = item.get('content', '')
            if role == 'user':
                history_formatted.append(f"User: {content}")
            elif role == 'assistant':
                history_formatted.append(f"Assistant: {content}")
        history_text = "Previous conversation:\n" + "\n".join(history_formatted)
        content_parts.append(history_text)
    
    # Add the question BEFORE the documents
    content_parts.append(f"Question: {question}")
    
    # Format document texts for document-based modes AFTER the question
    # Include chunks if: (1) category is in document_modes, OR (2) using iMedRAG retriever
    if category in document_modes or is_imedrag_retriever:
        if category == "pubmed" and pubmed_abstracts:
            # Handle PubMed content (reranked chunks or original abstracts)
            if pubmed_abstracts and 'content' in pubmed_abstracts[0]:
                # This is reranked chunks format
                document_texts = "\n\n".join([
                    f"---PubMed Content ({chunk['metadata']['pmid']}_{i})\n"
                    f"---short_citation: {chunk['metadata']['short_citation']}\n"
                    f"---title: {chunk['metadata']['title']}\n"
                    f"---authors: {', '.join(chunk['metadata']['authors'][:3])}{'...' if len(chunk['metadata']['authors']) > 3 else ''}\n"
                    f"---journal: {chunk['metadata']['journal']}\n"
                    f"---pub_date: {chunk['metadata']['pub_date']}\n"
                    f"---content_type: {'Full text' if chunk['metadata'].get('is_full_text') else 'Abstract only'}\n"
                    f"---similarity_score: {chunk['similarity_score']}\n"
                    f"{chunk['content']}"
                    for i, chunk in enumerate(pubmed_abstracts)
                ])
            else:
                # Fallback to original abstracts format
                document_texts = "\n\n".join([
                    f"---PubMed Abstract ({abstract['pmid']})\n"
                    f"---short_citation: {abstract['short_citation']}\n"
                    f"---title: {abstract['title']}\n"
                    f"---authors: {', '.join(abstract['authors'][:3])}{'...' if len(abstract['authors']) > 3 else ''}\n"
                    f"---journal: {abstract['journal']}\n"
                    f"---pub_date: {abstract['pub_date']}\n"
                    f"{abstract['abstract']}"
                    for abstract in pubmed_abstracts
                ])
            content_parts.append(document_texts)
        elif chunks:
            # Handle regular document chunks
            document_texts = "\n\n".join([
                f"---Document ({chunk.metadata['id']})\n"
                f"---short_citation: {chunk.metadata.get('short_citation', chunk.metadata.get('title', ''))}\n"
                f"{chunk.page_content}"
                for chunk in chunks
            ])
            content_parts.append(document_texts)
        
        # Add answer prompt for document modes
        content_parts.append("Answer:")
    else:
        # Add answer prompt for non-document modes
        content_parts.append("Answer:")
    
    return "\n\n".join(content_parts)


def create_prompt_messages(question, chunks=None, history=None, category=None, pubmed_abstracts=None, 
                           retriever_name=None, corpus_name=None, available_citations=None):
    """
    Create properly structured messages for the OpenAI API with separate system and user messages.
    Returns a list of messages in the format expected by the OpenAI API.
    """
    system_prompt = create_system_prompt(category, retriever_name=retriever_name, 
                                         corpus_name=corpus_name, available_citations=available_citations)
    user_content = create_user_content(question, chunks, history, category, pubmed_abstracts,
                                     retriever_name=retriever_name, corpus_name=corpus_name)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    return messages


def create_prompt(question, chunks=None, history=None, category=None, pubmed_abstracts=None,
                 retriever_name=None, corpus_name=None, available_citations=None):
    """
    Legacy function for backward compatibility.
    Creates a single combined prompt string as before.
    """
    messages = create_prompt_messages(question, chunks, history, category, pubmed_abstracts,
                                     retriever_name=retriever_name, corpus_name=corpus_name,
                                     available_citations=available_citations)
    # Combine system and user messages for backward compatibility
    return f"{messages[0]['content']}\n\n{messages[1]['content']}"

def synthesize(question, grounded, supplemental):
    prompt = ("Remove any information in the supplemental information section that has already been covered in "  
        "the grounded response. Do not return the grounded response. Rephrase any initiatial paragraphs in the "
        "supplemental information section so it\'s not claim to be the entire response. Try to format as many "
        "sections as bulleted lists or sections with headers using markdown as possible. \n\n"
        f"-Question: {question}\n\n"
        f"-Grounded response to the question: \n{grounded}\n\n"
        f"-Supplemental Information about the question:\n{supplemental}\n\n"
        f"-New supplemental information section:\n\n"
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return f'-**Response with citations**: \n\n{grounded}\n\n -**Supplemental information**\n\n{response.choices[0].message.content}'

def answer_question_with_citations(question, messages=None, prompt=None):
    """
    Answer the user's question using GPT-4o with inline citations.
    
    Args:
        question: The question being asked
        messages: Properly structured messages with system and user roles (preferred)
        prompt: Legacy single prompt string for backward compatibility
    """
    # Handle different input formats
    if messages is not None:
        # Use the new message structure (preferred)
        api_messages = messages
    elif prompt is not None:
        # Legacy single prompt - convert to old format for backward compatibility
        api_messages = [{"role": "user", "content": prompt}]
    else:
        # Fallback - treat question as simple user message
        api_messages = [{"role": "user", "content": question}]
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=api_messages,
        temperature=0.2
    )
    
    return response.choices[0].message.content


def _merge_and_rerank_chunks(question: str, all_retriever_results: List[Tuple[List, List]], 
                             reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                             top_k: int = 25) -> List[Tuple[Document, float]]:
    """
    Merge chunks from multiple retrievers, deduplicate, weight by retrieval count, and rerank.
    
    Args:
        question: The query question
        all_retriever_results: List of (chunks, scores) tuples from each retriever
        reranker_model_name: Name of the reranker model to use
        top_k: Number of top chunks to return after reranking
    
    Returns:
        List of (Document, rerank_score) tuples, sorted by rerank score (higher is better)
    """
    # Handle empty results
    if not all_retriever_results or all(len(chunks) == 0 for chunks, _ in all_retriever_results):
        logger.warning("No chunks retrieved from any retriever. Returning empty list.")
        return []
    
    # Step 1: Merge and deduplicate chunks, count retrieval frequency
    chunk_dict = {}  # chunk_id -> (Document, retrieval_count, original_scores)
    
    for chunks, scores in all_retriever_results:
        for chunk, score in zip(chunks, scores):
            chunk_id = chunk.metadata.get('id', '')
            if not chunk_id:
                # If no ID, use content hash as fallback
                import hashlib
                chunk_id = hashlib.md5(chunk.page_content.encode()).hexdigest()
                chunk.metadata['id'] = chunk_id
            
            if chunk_id in chunk_dict:
                # Chunk already seen - increment count and update scores
                doc, count, score_list = chunk_dict[chunk_id]
                chunk_dict[chunk_id] = (doc, count + 1, score_list + [score])
            else:
                # New chunk
                chunk_dict[chunk_id] = (chunk, 1, [score])
    
    # Step 2: Calculate weighted scores (weight = retrieval_count)
    weighted_chunks = []
    for chunk_id, (chunk, count, scores) in chunk_dict.items():
        # Average original scores and multiply by retrieval count as weight
        avg_score = sum(scores) / len(scores) if scores else 0.0
        weighted_score = avg_score * count  # Higher count = higher weight
        weighted_chunks.append((chunk, weighted_score, count))
    
    # Step 3: Rerank using cross-encoder reranker
    try:
        # Try to import FlagEmbedding (for BGE reranker)
        try:
            from FlagEmbedding import FlagReranker
            use_flag_reranker = True
        except ImportError:
            use_flag_reranker = False
        
        # Try to import sentence-transformers (for cross-encoder)
        try:
            from sentence_transformers import CrossEncoder
            use_cross_encoder = True
        except ImportError:
            use_cross_encoder = False
        
        if not use_flag_reranker and not use_cross_encoder:
            logger.warning("No reranker library found. Install FlagEmbedding or sentence-transformers. Using weighted scores only.")
            # Fallback: sort by weighted score
            weighted_chunks.sort(key=lambda x: x[1], reverse=True)
            return [(chunk, score) for chunk, score, _ in weighted_chunks[:top_k]]
        
        # Prepare pairs for reranking: (question, chunk_content)
        pairs = []
        chunk_list = []
        for chunk, weighted_score, count in weighted_chunks:
            # Combine title and content for reranking
            title = chunk.metadata.get('title', '')
            content = chunk.page_content
            if title:
                chunk_text = f"{title}. {content}"
            else:
                chunk_text = content
            pairs.append([question, chunk_text])
            chunk_list.append((chunk, weighted_score, count))
        
        # Rerank
        if use_flag_reranker and "bge" in reranker_model_name.lower():
            # Use FlagReranker (BGE models)
            reranker = FlagReranker(reranker_model_name, use_fp16=False)
            rerank_scores = reranker.compute_score(pairs)
            # Convert to list if single score
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
        elif use_cross_encoder:
            # Use CrossEncoder (sentence-transformers)
            reranker = CrossEncoder(reranker_model_name)
            rerank_scores = reranker.predict(pairs)
            # Convert to list if single score or numpy array
            import numpy as np
            if isinstance(rerank_scores, np.ndarray):
                rerank_scores = rerank_scores.tolist()
            elif not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
        else:
            # Fallback
            rerank_scores = [ws for _, ws, _ in chunk_list]
        
        # Combine rerank scores with retrieval count weight
        # Formula: final_score = rerank_score * (1 + log(retrieval_count))
        import math
        import numpy as np
        final_scores = []
        for i, (chunk, weighted_score, count) in enumerate(chunk_list):
            # Safely convert rerank_score to float
            rerank_score_val = rerank_scores[i]
            if isinstance(rerank_score_val, np.ndarray):
                # Handle numpy array (could be 0-d or 1-d)
                if rerank_score_val.ndim == 0:
                    rerank_score = float(rerank_score_val.item())
                else:
                    rerank_score = float(rerank_score_val.flatten()[0])
            else:
                rerank_score = float(rerank_score_val)
            
            # Apply retrieval count boost: log scale to avoid over-weighting
            count_boost = 1.0 + math.log(1 + count)  # log(1+count) so count=1 gives boost=1.0
            final_score = rerank_score * count_boost
            final_scores.append((chunk, final_score, count))
        
        # Sort by final score (higher is better)
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return [(chunk, score) for chunk, score, _ in final_scores[:top_k]]
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}. Falling back to weighted scores.")
        # Fallback: sort by weighted score
        weighted_chunks.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, score) for chunk, score, _ in weighted_chunks[:top_k]]


def get_rag_response(question, verbose=False, history=None, category=None, reasoning_mode=False, progress_callback=None, 
                    tot_depth=None, tot_branching_factor=None, tot_nodes_after_pruning=None, 
                    tot_model=None, tot_max_generation_tokens=None, tot_max_pruning_tokens=None,
                    prompt_example=None, tags=None, retriever_name=None, corpus_name="longcovid_clinical"):
    import time
    start_time = time.time()
    
    # Initialize debug data collection
    debug_data = {
        'question': question,
        'verbose': verbose,
        'category': category,
        'prompt_example': prompt_example,
        'reasoning_mode': reasoning_mode or 'standard',
        'history': history or [],
        'history_length': len(history) if history else 0,
        'chunks': [],
        'hyperparameters': {},
        'tot_data': None,
        'cache_status': 'unknown'
    }
    
    try:
        # Handle PubMed category separately
        if category == "pubmed":
            if PubMedService is None:
                raise ImportError("PubMedService is not available. Please ensure the pubmed_service module is accessible.")
            if progress_callback:
                progress_callback(10, "Initializing PubMed service...")
            
            # Initialize PubMed service
            pubmed_service = PubMedService(
                max_results=PUBMED_MAX_RESULTS,
                rate_limit_delay=PUBMED_RATE_LIMIT_DELAY
            )
            
            if progress_callback:
                progress_callback(20, "Searching PubMed...")
            
            # Get PubMed abstracts with full-text reranking
            pubmed_abstracts = pubmed_service.get_pubmed_abstracts(question, rerank_with_full_text=True)
            
            if not pubmed_abstracts:
                return {
                    "answer": "I couldn't find any relevant PubMed abstracts for your question. Please try rephrasing your question or using a different category.",
                    "chunks": []
                } if verbose else "I couldn't find any relevant PubMed abstracts for your question. Please try rephrasing your question or using a different category."
            
            # Store PubMed data for debug (now handles both abstracts and reranked chunks)
            debug_data['hyperparameters']['pubmed_items_retrieved'] = len(pubmed_abstracts)
            
            # Check if we have reranked chunks or original abstracts
            if pubmed_abstracts and 'content' in pubmed_abstracts[0]:
                # This is reranked chunks format
                debug_data['chunks'] = pubmed_abstracts
                
                # Log reranking statistics
                full_text_chunks = sum(1 for chunk in pubmed_abstracts 
                                     if chunk['metadata'].get('is_full_text', False))
                debug_data['hyperparameters']['full_text_chunks'] = full_text_chunks
                debug_data['hyperparameters']['abstract_only_chunks'] = len(pubmed_abstracts) - full_text_chunks
            else:
                # This is original abstracts format (fallback)
                debug_data['chunks'] = [
                    {
                        'content': abstract['abstract'],
                        'metadata': {
                            'pmid': abstract['pmid'],
                            'short_citation': abstract['short_citation'],
                            'title': abstract['title'],
                            'authors': abstract['authors'],
                            'journal': abstract['journal'],
                            'pub_date': abstract['pub_date'],
                            'source_type': 'pubmed',
                            'is_full_text': False,
                            'chunk_type': 'abstract'
                        },
                        'similarity_score': 'N/A - PubMed relevance'
                    }
                    for abstract in pubmed_abstracts
                ]
            
            if progress_callback:
                progress_callback(70, "Generating response from PubMed content...")
            
            # Check cache for PubMed responses
            cache = get_cache_instance()
            cached_answer = cache.get_cached_response(question, category, verbose)
            
            if cached_answer:
                print(f"Using cached PubMed response for question: {question[:50]}...")
                answer = cached_answer
                debug_data['cache_status'] = 'hit'
                # Still capture prompt for debugging even when cached
                messages = create_prompt_messages(question, [], history, category, pubmed_abstracts)
                debug_prompt = create_prompt(question, [], history, category, pubmed_abstracts)
                debug_data['generation_prompt'] = {
                    'type': 'pubmed',
                    'prompt': debug_prompt,
                    'messages': messages,
                    'cached': True
                }
            else:
                # Create messages with PubMed abstracts
                messages = create_prompt_messages(question, [], history, category, pubmed_abstracts)
                answer = answer_question_with_citations(question, messages)
                
                # Store prompt for debugging (create legacy prompt for debugging)
                debug_prompt = create_prompt(question, [], history, category, pubmed_abstracts)
                debug_data['generation_prompt'] = {
                    'type': 'pubmed',
                    'prompt': debug_prompt,
                    'messages': messages
                }
                
                # Store the response in cache
                cache.store_response(question, answer, category, verbose)
                debug_data['cache_status'] = 'miss'
            
            chunks = []  # Empty chunks for PubMed mode
            chunks_with_scores = []  # Empty chunks_with_scores for PubMed mode
            
        elif category == "web_search":
            # Handle web search category using unified create_prompt
            if progress_callback:
                progress_callback(20, "Performing web search...")
                
            if progress_callback:
                progress_callback(70, "Generating web search response...")
            
            # Check cache for web search responses
            cache = get_cache_instance()
            cached_answer = cache.get_cached_response(question, category, verbose)
            
            if cached_answer:
                print(f"Using cached web search response for question: {question[:50]}...")
                answer = cached_answer
                debug_data['cache_status'] = 'hit'
                # Still capture prompt for debugging even when cached
                messages = create_prompt_messages(question, chunks=None, history=history, category=category)
                debug_prompt = create_prompt(question, chunks=None, history=history, category=category)
                debug_data['generation_prompt'] = {
                    'type': 'web_search',
                    'prompt': debug_prompt,
                    'messages': messages,
                    'cached': True
                }
            else:
                messages = create_prompt_messages(question, chunks=None, history=history, category=category)
                answer = answer_question_with_citations(question, messages)
                
                # Store prompt for debugging (create legacy prompt for debugging)
                debug_prompt = create_prompt(question, chunks=None, history=history, category=category)
                debug_data['generation_prompt'] = {
                    'type': 'web_search',
                    'prompt': debug_prompt,
                    'messages': messages
                }
                
                # Store the response in cache
                cache.store_response(question, answer, category, verbose)
                debug_data['cache_status'] = 'miss'
            
            chunks = []  # Empty chunks for web search mode
            chunks_with_scores = []  # Empty chunks_with_scores for web search mode
            
        elif category == "no_retrieval":
            # Handle no retrieval category using unified create_prompt
            if progress_callback:
                progress_callback(20, "Preparing direct response...")
            
            if progress_callback:
                progress_callback(70, "Generating direct response...")
            
            # Check cache for no retrieval responses
            cache = get_cache_instance()
            cached_answer = cache.get_cached_response(question, category, verbose)
            
            if cached_answer:
                print(f"Using cached no retrieval response for question: {question[:50]}...")
                answer = cached_answer
                debug_data['cache_status'] = 'hit'
                # Still capture prompt for debugging even when cached
                messages = create_prompt_messages(question, chunks=None, history=history, category=category)
                debug_prompt = create_prompt(question, chunks=None, history=history, category=category)
                debug_data['generation_prompt'] = {
                    'type': 'no_retrieval',
                    'prompt': debug_prompt,
                    'messages': messages,
                    'cached': True
                }
            else:
                messages = create_prompt_messages(question, chunks=None, history=history, category=category)
                answer = answer_question_with_citations(question, messages)
                
                # Store prompt for debugging (create legacy prompt for debugging)
                debug_prompt = create_prompt(question, chunks=None, history=history, category=category)
                debug_data['generation_prompt'] = {
                    'type': 'no_retrieval',
                    'prompt': debug_prompt,
                    'messages': messages
                }
                
                # Store the response in cache
                cache.store_response(question, answer, category, verbose)
                debug_data['cache_status'] = 'miss'
            
            chunks = []  # Empty chunks for no retrieval mode
            chunks_with_scores = []  # Empty chunks_with_scores for no retrieval mode
            
        else:
            # Check if using iMedRAG retriever
            if retriever_name and retriever_name in ["BM25", "Contriever", "SPECTER", "MedCPT"]:
                # Use iMedRAG RetrievalSystem
                if progress_callback:
                    progress_callback(10, f"Initializing {retriever_name} retriever...")
                
                # Import iMedRAG RetrievalSystem
                import sys
                imedrag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "src"))
                if imedrag_path not in sys.path:
                    sys.path.insert(0, imedrag_path)
                from utils import RetrievalSystem
                
                # Initialize retrieval system
                # 强制使用 CPU 以避免 CUDA 内存问题
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 GPU
                retrieval_system = RetrievalSystem(
                    retriever_name=retriever_name,
                    corpus_name=corpus_name,
                    db_dir=os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "corpus"),
                    cache=False,
                    HNSW=False
                )
                
                if progress_callback:
                    progress_callback(20, f"Retrieving relevant documents using {retriever_name}...")
                
                target_chunks = 25
                
                # Retrieve chunks
                retrieved_snippets, scores = retrieval_system.retrieve(question, k=target_chunks, id_only=False)
                
                # Convert retrieved snippets to langchain Document format
                chunks_with_scores = []
                for snippet, score in zip(retrieved_snippets, scores):
                    # Convert snippet dict to Document
                    doc = Document(
                        page_content=snippet.get('content', ''),
                        metadata={
                            'id': snippet.get('id', ''),
                            'title': snippet.get('title', ''),
                            'short_citation': snippet.get('title', ''),  # Use title as short_citation
                            'categories': {}  # Empty categories for now
                        }
                    )
                    # For BM25, score is higher = better; for dense retrievers, score is similarity (higher = better)
                    # Normalize to distance-like format (lower = better) for consistency
                    if retriever_name == "BM25":
                        # BM25 scores are typically positive, higher is better
                        # Convert to distance-like (invert, but keep positive)
                        normalized_score = 1.0 / (1.0 + abs(score)) if score != 0 else 1.0
                    else:
                        # For dense retrievers, score might be similarity (higher = better) or distance (lower = better)
                        # Assuming similarity scores, convert to distance
                        normalized_score = 1.0 - score if score <= 1.0 else score
                    
                    chunks_with_scores.append((doc, normalized_score))
                
                # Sort by score (distance) in ascending order
                chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[1])
                
                # Use the sorted chunks
                chunks = [chunk for chunk, _ in chunks_with_scores]
                
                retrieval_attempts = [{
                    'k': target_chunks,
                    'total_retrieved': len(chunks),
                    'filtered_count': len(chunks)
                }]
                
            elif retriever_name == "OpenAIEmbedding":
                # Use OpenAI Embeddings + FAISS (original naive/naiverag.py approach)
                if progress_callback:
                    progress_callback(10, "Initializing OpenAI Embedding retriever...")
                
                vector_store = init_vector_store()
                
                if progress_callback:
                    progress_callback(20, "Retrieving relevant documents using OpenAI Embeddings...")
                
                target_chunks = 25
                retrieve_k = target_chunks
                chunks_with_scores = vector_store.similarity_search_with_score(question, k=retrieve_k)
                all_chunks_retrieved = len(chunks_with_scores)
                
                # Sort by L2 distance in ascending order (smallest distance = most similar first)
                chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[1])
                
                # Take only the target number of chunks (or all available if fewer)
                chunks_with_scores = chunks_with_scores[:target_chunks]
                
                # Use the sorted chunks
                chunks = [chunk for chunk, _ in chunks_with_scores]
                
                retrieval_attempts = [{
                    'k': retrieve_k,
                    'total_retrieved': all_chunks_retrieved,
                    'filtered_count': len(chunks_with_scores)
                }]
                
            elif retriever_name == "EnsembleRerank":
                # Sixth retriever: Merge results from all 5 retrievers, deduplicate, weight, and rerank
                if progress_callback:
                    progress_callback(10, "Initializing Ensemble Rerank retriever...")
                
                # List of all base retrievers
                base_retrievers = ["BM25", "Contriever", "SPECTER", "MedCPT", "OpenAIEmbedding"]
                
                # Collect results from all base retrievers
                all_retriever_results = []
                retrieval_stats = {}
                
                for base_retriever in base_retrievers:
                    if progress_callback:
                        progress_callback(20, f"Retrieving with {base_retriever}...")
                    
                    try:
                        # Temporarily set retriever_name to base retriever
                        # We'll call get_rag_response recursively but only get chunks, not generate answer
                        # Actually, better to directly call retrieval logic
                        import sys
                        imedrag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "src"))
                        if imedrag_path not in sys.path:
                            sys.path.insert(0, imedrag_path)
                        
                        if base_retriever == "OpenAIEmbedding":
                            # Use OpenAI Embeddings
                            vector_store = init_vector_store()
                            target_chunks = 25
                            chunks_with_scores = vector_store.similarity_search_with_score(question, k=target_chunks)
                            chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[1])
                            chunks = [chunk for chunk, _ in chunks_with_scores]
                            scores = [score for _, score in chunks_with_scores]
                        else:
                            # Use iMedRAG RetrievalSystem
                            from utils import RetrievalSystem
                            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
                            retrieval_system = RetrievalSystem(
                                retriever_name=base_retriever,
                                corpus_name=corpus_name,
                                db_dir=os.path.join(os.path.dirname(__file__), "..", "iMedRAG", "corpus"),
                                cache=False,
                                HNSW=False
                            )
                            target_chunks = 25
                            retrieved_snippets, scores = retrieval_system.retrieve(question, k=target_chunks, id_only=False)
                            
                            # Convert to Document format
                            chunks = []
                            for snippet in retrieved_snippets:
                                doc = Document(
                                    page_content=snippet.get('content', ''),
                                    metadata={
                                        'id': snippet.get('id', ''),
                                        'title': snippet.get('title', ''),
                                        'short_citation': snippet.get('title', ''),
                                        'categories': {}
                                    }
                                )
                                chunks.append(doc)
                        
                        all_retriever_results.append((chunks, scores))
                        retrieval_stats[base_retriever] = len(chunks)
                        
                    except Exception as e:
                        logger.warning(f"Error retrieving with {base_retriever}: {e}. Skipping.")
                        retrieval_stats[base_retriever] = 0
                        continue
                
                if progress_callback:
                    progress_callback(50, "Merging and deduplicating chunks...")
                
                # Merge, deduplicate, weight, and rerank
                if progress_callback:
                    progress_callback(60, "Reranking chunks...")
                
                reranked_chunks_with_scores = _merge_and_rerank_chunks(
                    question=question,
                    all_retriever_results=all_retriever_results,
                    reranker_model_name="cross-encoder/ms-marco-electra-base",  # Using different CrossEncoder model
                    top_k=25
                )
                
                # Extract chunks and scores
                chunks = [chunk for chunk, _ in reranked_chunks_with_scores]
                scores = [score for _, score in reranked_chunks_with_scores]
                
                # Create chunks_with_scores for consistency with other retrievers
                chunks_with_scores = list(zip(chunks, scores))
                
                retrieval_attempts = [{
                    'k': 25,
                    'total_retrieved': sum(retrieval_stats.values()),
                    'filtered_count': len(chunks),
                    'base_retrievers': retrieval_stats
                }]
                
            else:
                # Regular vector store handling (original OpenAI Embeddings + FAISS)
                if progress_callback:
                    progress_callback(10, "Initializing vector store...")
                
                vector_store = init_vector_store()
            
            # Log the selected category
            if category:
                print(f"Processing question with category: {category}")
            
            if progress_callback:
                progress_callback(20, "Retrieving relevant documents...")
            
            if progress_callback:
                progress_callback(40, "Processing document chunks...")
            
            # Iterative retrieval to guarantee target chunk count
            target_chunks = 25
            # Get total chunks in database by using a very high k value to retrieve all
            max_k = vector_store.index.ntotal if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal') else 10000
            filtered_chunks_with_scores = []
            all_chunks_retrieved = 0
            retrieval_attempts = []
            
            if category and category != "pubmed":
                # For category filtering, use iterative approach to get target_chunks
                k_values = [25, 50, 100, 200, 400, 800, 1200, 2000, max_k]
                rejected_docs = set()  # Track rejected document IDs for debugging
                
                for retrieve_k in k_values:
                    logger.info(f"Retrieving {retrieve_k} chunks for category '{category}' (attempt {len(retrieval_attempts) + 1})")
                    
                    chunks_with_scores = vector_store.similarity_search_with_score(question, k=retrieve_k)
                    all_chunks_retrieved = len(chunks_with_scores)
                    
                    # Sort by L2 distance in ascending order (smallest distance = most similar first)
                    chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[1])
                    
                    # Filter by category
                    filtered_chunks_with_scores = []
                    for chunk, score in chunks_with_scores:
                        # Get the document's categories from metadata
                        chunk_categories = chunk.metadata.get('categories', {})
                        doc_id = chunk.metadata.get('id', '')
                        
                        # Ensure we're checking a boolean value from the dictionary
                        if isinstance(chunk_categories, dict):
                            # Check if the category key exists and is explicitly True (not None, not False)
                            category_value = chunk_categories.get(category)
                            if category_value is True:
                                filtered_chunks_with_scores.append((chunk, score))
                            else:
                                rejected_docs.add(doc_id)
                        # Fallback for older index format (if categories is a list)
                        elif isinstance(chunk_categories, list) and category in chunk_categories:
                            filtered_chunks_with_scores.append((chunk, score))
                        else:
                            rejected_docs.add(doc_id)
                    
                    retrieval_attempts.append({
                        'k': retrieve_k,
                        'total_retrieved': all_chunks_retrieved,
                        'filtered_count': len(filtered_chunks_with_scores)
                    })
                    
                    logger.info(f"Retrieved {all_chunks_retrieved} total chunks, {len(filtered_chunks_with_scores)} match category '{category}'")
                    
                    # Check if we have enough chunks or reached maximum
                    if len(filtered_chunks_with_scores) >= target_chunks or retrieve_k >= max_k:
                        break
                
                # Take the top target_chunks filtered chunks (or all if fewer than target available)
                chunks_with_scores = filtered_chunks_with_scores[:target_chunks]
                
                # Log retrieval summary
                final_chunk_count = len(chunks_with_scores)
                logger.info(f"Final retrieval for category '{category}': {final_chunk_count} chunks after {len(retrieval_attempts)} attempts")
                
                # Only log rejected documents if they were unexpected
                if 'Fineberg' in rejected_docs or any('NASEM' in doc for doc in rejected_docs):
                    logger.info(f"Filtered out {len(rejected_docs)} documents including: {', '.join(sorted(list(rejected_docs)[:5]))}")
                    
            else:
                # No category filtering, use simple retrieval
                retrieve_k = target_chunks
                chunks_with_scores = vector_store.similarity_search_with_score(question, k=retrieve_k)
                all_chunks_retrieved = len(chunks_with_scores)
                
                # Sort by L2 distance in ascending order (smallest distance = most similar first)
                chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[1])
                
                # Take only the target number of chunks (or all available if fewer)
                chunks_with_scores = chunks_with_scores[:target_chunks]
                
                retrieval_attempts = [{
                    'k': retrieve_k,
                    'total_retrieved': all_chunks_retrieved,
                    'filtered_count': len(chunks_with_scores)
                }]
            
            # Use the sorted chunks for creating the prompt
            chunks = [chunk for chunk, _ in chunks_with_scores]
            
            # Store retrieval statistics for debug (for both iMedRAG and regular retrieval)
            debug_data['hyperparameters']['final_chunks_used'] = len(chunks)
            debug_data['hyperparameters']['target_chunks'] = target_chunks
            if 'retrieval_attempts' in locals():
                debug_data['hyperparameters']['retrieval_attempts'] = retrieval_attempts
                debug_data['hyperparameters']['total_retrieval_attempts'] = len(retrieval_attempts)
            if retriever_name and retriever_name in ["BM25", "Contriever", "SPECTER", "MedCPT", "OpenAIEmbedding"]:
                debug_data['hyperparameters']['retriever_name'] = retriever_name
                debug_data['hyperparameters']['corpus_name'] = corpus_name
            
            # Prepare chunks data for debug output
            # Ensure chunks_with_scores is defined (should be set in all retrieval paths)
            if 'chunks_with_scores' not in locals():
                chunks_with_scores = []
            
            debug_data['chunks'] = [
                {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'similarity_score': float(score)
                }
                for chunk, score in chunks_with_scores
            ]
            
            # Check if we have enough chunks
            if len(chunks) == 0:
                return {
                    "answer": "I couldn't find any relevant information in the selected document category. Please try a different category or question.",
                    "chunks": []
                } if verbose else "I couldn't find any relevant information in the selected document category. Please try a different category or question."
            
            if progress_callback:
                progress_callback(70, "Generating response...")
            
            # Generate available citations dynamically from retrieved chunks
            available_citations = None
            if retriever_name and retriever_name in ["BM25", "Contriever", "SPECTER", "MedCPT"] and corpus_name in ["longcovid_clinical", "longcovid_expanded"]:
                # Extract unique titles from chunks
                available_citations = sorted(list(set([chunk.metadata.get('title', '') for chunk in chunks if chunk.metadata.get('title')])))
            elif retriever_name == "OpenAIEmbedding":
                # For OpenAIEmbedding, extract chunk IDs from metadata
                available_citations = sorted(list(set([chunk.metadata.get('id', '') for chunk in chunks if chunk.metadata.get('id')])))
            elif retriever_name == "EnsembleRerank":
                # For EnsembleRerank, extract chunk IDs from metadata (same as OpenAIEmbedding)
                available_citations = sorted(list(set([chunk.metadata.get('id', '') for chunk in chunks if chunk.metadata.get('id')])))
            
            # Check cache for base response first
            # For EnsembleRerank, disable cache to ensure each question gets a unique answer
            cache = get_cache_instance()
            if retriever_name == "EnsembleRerank":
                # Disable cache for EnsembleRerank to ensure different answers for different questions
                cached_answer = None
            else:
                cached_answer = cache.get_cached_response(question, category, verbose) if cache.is_enabled() else None
            
            if cached_answer:
                print(f"Using cached response for question: {question[:50]}...")
                answer = cached_answer
                debug_data['cache_status'] = 'hit'
                # Still capture prompt for debugging even when cached
                messages = create_prompt_messages(question, chunks, history, category,
                                                 retriever_name=retriever_name, corpus_name=corpus_name,
                                                 available_citations=available_citations)
                debug_prompt = create_prompt(question, chunks, history, category,
                                            retriever_name=retriever_name, corpus_name=corpus_name,
                                            available_citations=available_citations)
                debug_data['generation_prompt'] = {
                    'type': 'rag',
                    'category': category,
                    'prompt': debug_prompt,
                    'messages': messages,
                    'cached': True
                }
            else:
                # Create messages for RAG response
                messages = create_prompt_messages(question, chunks, history, category,
                                                 retriever_name=retriever_name, corpus_name=corpus_name,
                                                 available_citations=available_citations)
                answer = answer_question_with_citations(question, messages)
                
                # Store prompt for debugging (create legacy prompt for debugging)
                debug_prompt = create_prompt(question, chunks, history, category,
                                            retriever_name=retriever_name, corpus_name=corpus_name,
                                            available_citations=available_citations)
                debug_data['generation_prompt'] = {
                    'type': 'rag',
                    'category': category,
                    'prompt': debug_prompt,
                    'messages': messages
                }
                
                # Store the response in cache
                cache.store_response(question, answer, category, verbose)
                debug_data['cache_status'] = 'miss'
        
        # Add reasoning sections based on reasoning mode
        tot_result = None  # Initialize for verbose mode
        if reasoning_mode == "tree_of_thoughts":
            if progress_callback:
                progress_callback(75, "Initiating Tree of Thoughts reasoning...")
            
            tot_start_time = time.time()
            # Execute Tree of Thoughts with provided parameters or defaults
            actual_tot_depth = tot_depth if tot_depth is not None else TOT_DEFAULT_DEPTH
            actual_tot_branching_factor = tot_branching_factor if tot_branching_factor is not None else TOT_DEFAULT_BRANCHING_FACTOR
            actual_tot_nodes_after_pruning = tot_nodes_after_pruning if tot_nodes_after_pruning is not None else TOT_DEFAULT_NODES_AFTER_PRUNING
            
            tot_result = execute_tree_of_thoughts(
                question, 
                answer, 
                depth=actual_tot_depth,
                branching_factor=actual_tot_branching_factor,
                nodes_after_pruning=actual_tot_nodes_after_pruning,
                prompt_example=prompt_example,
                progress_callback=progress_callback
            )
            tot_end_time = time.time()
            
            # Store ToT data for debug
            debug_data['tot_data'] = tot_result
            debug_data['hyperparameters']['tot_processing_time'] = round(tot_end_time - tot_start_time, 2)
            debug_data['hyperparameters']['tot_depth'] = actual_tot_depth
            debug_data['hyperparameters']['tot_branching_factor'] = actual_tot_branching_factor
            debug_data['hyperparameters']['tot_nodes_after_pruning'] = actual_tot_nodes_after_pruning
            
            # Add the follow-up questions from ToT
            answer = f"{answer}\n\n## Reasoning Insights\n\n{tot_result['follow_up_questions']}"
        
        # Complete hyperparameters for debug
        end_time = time.time()
        debug_data['hyperparameters'].update({
            'processing_time': round(end_time - start_time, 2),
            'chunk_size': 1200,
            'chunk_overlap': 600,
            'main_model': 'gpt-4o',
            'main_temperature': 0.2,
            'tot_model': 'gpt-4o-mini',
            'tot_max_generation_tokens': 200,
            'tot_max_pruning_tokens': 200,
            'similarity_method': 'L2 distance',
            'cache_enabled': cache.is_enabled() if cache else False
        })
        
        # Generate debug file if enabled and response output file
        try:
            debug_manager = get_debug_manager()
            debug_file_path = None
            
            if debug_manager.is_enabled():
                debug_file_path = debug_manager.generate_debug_file(
                    question=question,
                    response=answer,
                    chunks=debug_data['chunks'],
                    hyperparameters=debug_data['hyperparameters'],
                    history=history,
                    tot_data=debug_data['tot_data'],
                    generation_prompt=debug_data.get('generation_prompt')
                )
            
            # Always generate response output file
            response_manager = get_response_manager()
            debug_filename = os.path.basename(debug_file_path) if debug_file_path else None
            logger.info(f"Generating response output file for question: {question[:50]}...")
            response_file_path = response_manager.generate_response_file(
                question=question,
                response=answer,
                chunks=debug_data['chunks'],
                debug_filename=debug_filename,
                category=category,
                metadata={
                    'verbose_mode': verbose,
                    'history_length': len(history) if history else 0,
                    'reasoning_mode': reasoning_mode or 'standard'
                },
                tags=tags
            )
            if response_file_path:
                logger.info(f"Successfully generated response file: {response_file_path}")
            else:
                logger.warning("Failed to generate response output file")
        except Exception as e:
            logger.error(f"Failed to generate output files: {e}")
        
        if verbose:
            # Handle different category types
            if category in ["pubmed", "web_search", "no_retrieval"]:
                response_data = {
                    "answer": answer,
                    "chunks": debug_data['chunks'] if category == "pubmed" else []  # Only PubMed has chunks in debug_data
                }
            else:
                response_data = {
                    "answer": answer,
                    "chunks": [
                        {
                            "content": chunk.page_content,
                            "metadata": chunk.metadata,
                            "similarity_score": float(score)  # This is now the L2 distance - smaller means more similar
                        }
                        for chunk, score in chunks_with_scores
                    ]
                }
            
            # Add ToT data if it was executed
            if reasoning_mode == "tree_of_thoughts" and tot_result:
                response_data["tree_of_thoughts"] = tot_result
            
            return response_data
        return answer
    except Exception as e:
        print(f"Error in get_rag_response: {str(e)}")
        raise

def synthesized_response(question, verbose=False, history=None, category=None, reasoning_mode=False, progress_callback=None, 
                        tot_depth=None, tot_branching_factor=None, tot_nodes_after_pruning=None, 
                        tot_model=None, tot_max_generation_tokens=None, tot_max_pruning_tokens=None,
                        prompt_example=None):
    if verbose:
        return get_rag_response(question, verbose=True, history=history, category=category, reasoning_mode=reasoning_mode, progress_callback=progress_callback,
                               tot_depth=tot_depth, tot_branching_factor=tot_branching_factor, tot_nodes_after_pruning=tot_nodes_after_pruning,
                               tot_model=tot_model, tot_max_generation_tokens=tot_max_generation_tokens, tot_max_pruning_tokens=tot_max_pruning_tokens,
                               prompt_example=prompt_example)
    return get_rag_response(question, verbose=False, history=history, category=category, reasoning_mode=reasoning_mode, progress_callback=progress_callback,
                           tot_depth=tot_depth, tot_branching_factor=tot_branching_factor, tot_nodes_after_pruning=tot_nodes_after_pruning,
                           tot_model=tot_model, tot_max_generation_tokens=tot_max_generation_tokens, tot_max_pruning_tokens=tot_max_pruning_tokens,
                           prompt_example=prompt_example)

if __name__ == "__main__":
    question = "What are the main symptoms of long COVID?"
    print(synthesized_response(question))

    
