import os
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dirname = os.path.dirname(__file__)
CORPUS_NAME = "LongCovid"

def load_documents(folder_name=CORPUS_NAME, index_csv_name=f'{CORPUS_NAME}.csv', category=None, only_included=False):
    """
    Load documents from the corpus folder.
    
    Args:
        folder_name: The folder containing the documents
        index_csv_name: The CSV file with document metadata
        category: Optional filter for document category ('clinical_only', 'clinical_expanded', 'clinical_and_research')
                 If None, all documents are loaded
        only_included: If True, only load documents with 'include' = 'y'
    
    Returns:
        List of documents with their metadata
    """
    corpus_dir = os.path.join(dirname,'corpus')
    article_cache_dir = os.path.join(corpus_dir, 'txt', folder_name)
    index_csv_path = os.path.join(corpus_dir, 'csv', index_csv_name)
    
    logger.info(f"Loading documents from {index_csv_path}")
    logger.info(f"Text files directory: {article_cache_dir}")
    
    docs = []
    ids = []
    short_citations = []
    categories = []
    
    # Count for logging
    total_rows = 0
    included_rows = 0
    excluded_no_include = 0
    excluded_category_filter = 0
    excluded_missing_file = 0
    
    with open(index_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1
            
            # Skip if not included in the corpus and only_included is True
            include_value = row.get('include', '').strip()
            if only_included and include_value != 'y':
                excluded_no_include += 1
                continue
                
            # Check if document belongs to the requested category
            if category:
                category_value = row.get(category, '').strip()
                if category_value != 'y':
                    excluded_category_filter += 1
                    continue
            
            txt_name = row.get('file_name')
            short_citation = row.get('short_citation')
            
            # Skip if no file name
            if not txt_name:
                excluded_missing_file += 1
                continue
            
            # Create a categories dict for this document
            doc_categories = {
                'clinical_only': row.get('clinical_only', '').strip() == 'y',
                'clinical_expanded': row.get('clinical_expanded', '').strip() == 'y',
                'clinical_and_research': row.get('clinical_and_research', '').strip() == 'y',
                'compendium_references': row.get('compendium_references', '').strip() == 'y',
                'compendium_with_references': row.get('compendium_with_references', '').strip() == 'y'
            }
            
            txt_path = os.path.join(article_cache_dir, txt_name + '.txt')
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    docs.append(f.read())
                    ids.append(txt_name)
                    short_citations.append(short_citation)
                    categories.append(doc_categories)
                    included_rows += 1
            except FileNotFoundError:
                logger.warning(f"File not found: {txt_path}")
                excluded_missing_file += 1
                continue
    
    # Log document loading statistics
    logger.info(f"CSV total rows: {total_rows}")
    if only_included:
        logger.info(f"Excluded - no 'include=y': {excluded_no_include}")
    if category:
        logger.info(f"Excluded - not in category '{category}': {excluded_category_filter}")
    logger.info(f"Excluded - missing text file: {excluded_missing_file}")
    logger.info(f"Documents loaded: {included_rows}")
    
    return [{
        "content": doc, 
        "id": doc_id, 
        "short_citation": short_citation,
        "categories": category_info
    } for doc, doc_id, short_citation, category_info in zip(docs, ids, short_citations, categories)]

def get_document_categories():
    """Return all available document categories"""
    return {
        "clinical_only": "Clinical only",
        "clinical_expanded": "Clinical expanded",
        "clinical_and_research": "Clinical & Research",
        "compendium_references": "Compendium references",
        "compendium_with_references": "Compendium with references",
        "pubmed": "PubMed abstracts",
        "web_search": "Web search",
        "no_retrieval": "No retrieval"
    }
