"""
Agentic RAG module for Unsiloed AI with hierarchical chunking
- Supports parent-child relationships: page -> section -> semantic chunk
"""
from typing import List, Dict, Optional

class HierarchicalChunk:
    def __init__(self, text: str, chunk_id: str, parent_id: Optional[str] = None, level: str = "semantic", metadata: dict = None):
        self.text = text
        self.chunk_id = chunk_id
        self.parent_id = parent_id  # e.g., page or section id
        self.level = level  # 'semantic', 'section', 'page', etc.
        self.metadata = metadata or {}

class SimpleVectorStore:
    def __init__(self):
        self.chunks = {}  # chunk_id -> HierarchicalChunk
        self.vectors = []  # List of (embedding, chunk_id)

    def add(self, embedding, chunk: HierarchicalChunk):
        self.chunks[chunk.chunk_id] = chunk
        self.vectors.append((embedding, chunk.chunk_id))

    def search(self, query_embedding, top_k=3):
        # Dummy similarity: returns first k semantic chunks
        semantic_chunks = [cid for _, cid in self.vectors if self.chunks[cid].level == "semantic"]
        return [self.chunks[cid] for cid in semantic_chunks[:top_k]]

    def get_parent(self, chunk: HierarchicalChunk):
        if chunk.parent_id:
            return self.chunks.get(chunk.parent_id)
        return None

def embed_text(text: str) -> List[float]:
    # Dummy embedding for demonstration
    return [float(ord(c)) for c in text[:10]]

class AgenticRAG:
    def __init__(self):
        self.vector_store = SimpleVectorStore()

    def index_hierarchical_chunks(self, pages: List[dict]):
        """
        pages: List of dicts, each with keys:
            - 'page_id': str
            - 'text': str (page text)
            - 'sections': List[dict] with keys 'section_id', 'text', 'semantic_chunks'
        """
        for page in pages:
            page_chunk = HierarchicalChunk(
                text=page['text'],
                chunk_id=page['page_id'],
                parent_id=None,
                level="page"
            )
            self.vector_store.chunks[page['page_id']] = page_chunk
            for section in page.get('sections', []):
                section_chunk = HierarchicalChunk(
                    text=section['text'],
                    chunk_id=section['section_id'],
                    parent_id=page['page_id'],
                    level="section"
                )
                self.vector_store.chunks[section['section_id']] = section_chunk
                for sem in section.get('semantic_chunks', []):
                    sem_chunk = HierarchicalChunk(
                        text=sem['text'],
                        chunk_id=sem['chunk_id'],
                        parent_id=section['section_id'],
                        level="semantic"
                    )
                    emb = embed_text(sem['text'])
                    self.vector_store.add(emb, sem_chunk)

    def retrieve(self, query: str, top_k=3) -> List[HierarchicalChunk]:
        query_emb = embed_text(query)
        return self.vector_store.search(query_emb, top_k=top_k)

    def get_hierarchical_context(self, chunks: List[HierarchicalChunk]) -> str:
        context = []
        for chunk in chunks:
            section = self.vector_store.get_parent(chunk)
            page = self.vector_store.get_parent(section) if section else None
            if page:
                context.append(f"[Page: {page.chunk_id}] {page.text[:100]}...")
            if section:
                context.append(f"[Section: {section.chunk_id}] {section.text[:100]}...")
            context.append(f"[Semantic: {chunk.chunk_id}] {chunk.text}")
        return "\n".join(context)

    def generate_answer(self, query: str, context_chunks: List[HierarchicalChunk]) -> str:
        context = self.get_hierarchical_context(context_chunks)
        return f"Q: {query}\nContext:\n{context}\nA: [LLM output here]"

    def run_hierarchical(self, query: str, pages: List[dict], top_k=3) -> str:
        self.index_hierarchical_chunks(pages)
        context_chunks = self.retrieve(query, top_k=top_k)
        return self.generate_answer(query, context_chunks)
