# rag_engine.py
# Advanced RAG Engine with Hybrid Search, Reranking, and Query Transformation

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results with scores"""
    document: Document
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class AdvancedRAGEngine:
    """
    Advanced RAG Engine ที่รวม:
    1. Hybrid Search (BM25 + Vector)
    2. Reranking (Cross-Encoder)
    3. Query Transformation (Multi-Query)
    """
    
    def __init__(
        self, 
        vectorstore: FAISS,
        llm = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        use_reranker: bool = True,
        use_multi_query: bool = False
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.use_reranker = use_reranker
        self.use_multi_query = use_multi_query
        
        # Initialize BM25 index from vectorstore documents
        self._init_bm25_index()
        
        # Initialize reranker (lazy loading)
        self._reranker = None
    
    def _init_bm25_index(self):
        """Extract documents from FAISS and build BM25 index"""
        try:
            # Get all documents from FAISS
            self.all_docs = []
            docstore = self.vectorstore.docstore
            
            for doc_id in self.vectorstore.index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                if doc:
                    self.all_docs.append(doc)
            
            # Tokenize for BM25
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.all_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            logger.info(f"BM25 index built with {len(self.all_docs)} documents")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
            self.all_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25 - works for both Thai and English"""
        # Basic word splitting (works reasonably for both languages)
        import re
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _get_reranker(self):
        """Lazy load reranker model"""
        if self._reranker is None and self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                # Use a lightweight, multilingual-friendly model
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-Encoder reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self._reranker = False  # Mark as unavailable
        return self._reranker if self._reranker else None
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 4,
        language: Optional[str] = None,
        fetch_k: int = 20
    ) -> List[Document]:
        """
        Hybrid Search: รวม BM25 (keyword) + Vector (semantic)
        ใช้ Reciprocal Rank Fusion (RRF) เพื่อรวมผลลัพธ์
        """
        results: Dict[str, SearchResult] = {}
        
        # 1. Vector Search
        try:
            filter_dict = {"language": language} if language else None
            vector_results = self.vectorstore.similarity_search_with_score(
                query, k=fetch_k, filter=filter_dict
            )
            
            for rank, (doc, score) in enumerate(vector_results):
                doc_id = hash(doc.page_content[:200])  # Use content hash as ID
                if doc_id not in results:
                    results[doc_id] = SearchResult(document=doc)
                # RRF score: 1 / (rank + 60)
                results[doc_id].vector_score = 1.0 / (rank + 60)
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # 2. BM25 Search
        if self.bm25 is not None:
            try:
                query_tokens = self._tokenize(query)
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top-k indices
                top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
                
                for rank, idx in enumerate(top_indices):
                    if bm25_scores[idx] > 0:  # Only include if score > 0
                        doc = self.all_docs[idx]
                        
                        # Apply language filter
                        if language and doc.metadata.get("language") != language:
                            continue
                        
                        doc_id = hash(doc.page_content[:200])
                        if doc_id not in results:
                            results[doc_id] = SearchResult(document=doc)
                        # RRF score
                        results[doc_id].bm25_score = 1.0 / (rank + 60)
                        
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
        
        # 3. Combine scores using weighted RRF
        for result in results.values():
            result.final_score = (
                self.bm25_weight * result.bm25_score + 
                self.vector_weight * result.vector_score
            )
        
        # 4. Sort by final score and get top-k
        sorted_results = sorted(
            results.values(), 
            key=lambda x: x.final_score, 
            reverse=True
        )[:fetch_k]
        
        # 5. Rerank if enabled
        if self.use_reranker and len(sorted_results) > 0:
            sorted_results = self._rerank(query, sorted_results, k)
        
        # Return top-k documents
        return [r.document for r in sorted_results[:k]]
    
    def _rerank(
        self, 
        query: str, 
        results: List[SearchResult],
        k: int
    ) -> List[SearchResult]:
        """Rerank results using Cross-Encoder"""
        reranker = self._get_reranker()
        if not reranker:
            return results
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, r.document.page_content] for r in results]
            
            # Get rerank scores
            scores = reranker.predict(pairs)
            
            # Update scores
            for i, score in enumerate(scores):
                results[i].rerank_score = float(score)
                results[i].final_score = float(score)  # Override with rerank score
            
            # Re-sort by rerank score
            return sorted(results, key=lambda x: x.final_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
    
    def search_with_multi_query(
        self, 
        query: str, 
        k: int = 4,
        language: Optional[str] = None
    ) -> List[Document]:
        """
        Multi-Query: สร้างหลาย variations ของคำถามแล้ว search
        """
        if not self.use_multi_query or not self.llm:
            return self.hybrid_search(query, k, language)
        
        try:
            # Generate query variations using LLM
            variations = self._generate_query_variations(query, language)
            
            # Search with all variations
            all_results: Dict[str, Document] = {}
            
            for q in [query] + variations:
                docs = self.hybrid_search(q, k=k*2, language=language)
                for doc in docs:
                    doc_id = hash(doc.page_content[:200])
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
            
            # Return top-k unique results
            return list(all_results.values())[:k]
            
        except Exception as e:
            logger.error(f"Multi-query search failed: {e}")
            return self.hybrid_search(query, k, language)
    
    def _generate_query_variations(self, query: str, language: str) -> List[str]:
        """Generate query variations using LLM"""
        if not self.llm:
            return []
        
        try:
            if language == "th":
                prompt = f"""สร้างคำถาม 3 แบบที่มีความหมายใกล้เคียงกับคำถามนี้:
คำถาม: {query}

ตอบเป็น list แบบนี้:
1. [คำถามแบบที่ 1]
2. [คำถามแบบที่ 2]
3. [คำถามแบบที่ 3]"""
            else:
                prompt = f"""Generate 3 alternative phrasings for this question:
Question: {query}

Answer as a list:
1. [Alternative 1]
2. [Alternative 2]
3. [Alternative 3]"""
            
            response = self.llm.invoke(prompt)
            
            # Parse response to extract variations
            import re
            variations = re.findall(r'\d+\.\s*(.+)', response.content)
            return variations[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate query variations: {e}")
            return []


# Convenience function for backward compatibility
def create_rag_engine(
    vectorstore: FAISS,
    llm = None,
    use_reranker: bool = True,
    use_multi_query: bool = False
) -> AdvancedRAGEngine:
    """Factory function to create RAG engine"""
    return AdvancedRAGEngine(
        vectorstore=vectorstore,
        llm=llm,
        use_reranker=use_reranker,
        use_multi_query=use_multi_query
    )
