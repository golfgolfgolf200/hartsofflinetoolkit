import re
import numpy as np
import pyarrow as pa
import torch
import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import settings, apply_hardware_governor

apply_hardware_governor()

class SearchEngine:
    def __init__(self):
        print(f"Loading local weights: {settings.model_path} on CPU (float32)...")
        self.model = SentenceTransformer(
            settings.model_path,
            device="cpu",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float32}
        )
        self.dim = settings.matryoshka_dim
        
        self.db = lancedb.connect(settings.db_path)
        self.schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.dim)),
            pa.field("doc_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("content", pa.string()),
        ])
        self.table = self.db.create_table("documents", schema=self.schema, exist_ok=True)
        
        self.corpus_ids = []
        self.corpus_docs = []
        self.bm25 = None
        self._rebuild_bm25_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _rebuild_bm25_index(self):
        records = self.table.to_pandas()
        if not records.empty:
            self.corpus_ids = records["doc_id"].tolist()
            self.corpus_docs = records.to_dict("records")
            tokenized = [self._tokenize(doc["content"]) for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.corpus_ids = []
            self.corpus_docs = []
            self.bm25 = None

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        formatted = f"task: search result | query: {text}" if is_query else f"title: none | text: {text}"
        full_vector = self.model.encode(formatted, normalize_embeddings=False)
        truncated = full_vector[:self.dim]
        norm = np.linalg.norm(truncated)
        if norm > 0:
            truncated = truncated / norm
        return truncated.tolist()

    def index_documents(self, docs: list[dict]):
        if not docs:
            return 0
        records = []
        for d in docs:
            vec = self.embed_text(d["content"], is_query=False)
            records.append({
                "vector": vec,
                "doc_id": d["doc_id"],
                "source": d["source"],
                "page": d["page"],
                "content": d["content"],
            })
        self.table.add(records)
        self._rebuild_bm25_index()
        return len(records)

    def hybrid_search(self, query: str, top_n: int = 5) -> list[dict]:
        if not self.bm25 or len(self.corpus_ids) == 0:
            return []

        tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        
        bm25_ranks = {}
        for rank, idx in enumerate(bm25_ranked):
            if bm25_scores[idx] > 0:
                bm25_ranks[self.corpus_ids[idx]] = rank + 1

        q_vec = self.embed_text(query, is_query=True)
        dense_results = self.table.search(q_vec).limit(len(self.corpus_ids)).to_list()
        
        dense_ranks = {}
        dense_doc_map = {}
        for rank, res in enumerate(dense_results):
            doc_id = res["doc_id"]
            dense_ranks[doc_id] = rank + 1
            dense_doc_map[doc_id] = res

        all_candidate_ids = set(bm25_ranks.keys()).union(set(dense_ranks.keys()))
        fused = []
        k = settings.rrf_k
        for doc_id in all_candidate_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[doc_id])
            
            doc_data = dense_doc_map.get(doc_id) or next(d for d in self.corpus_docs if d["doc_id"] == doc_id)
            fused.append({
                "doc_id": doc_id,
                "score": score,
                "source": doc_data["source"],
                "page": doc_data["page"],
                "content": doc_data["content"]
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_n]
