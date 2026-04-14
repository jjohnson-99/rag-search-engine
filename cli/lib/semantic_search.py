import json
import os
import re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_DIR, load_movies

SCORE_PRECISION = 4

class SemanticSearch():
    def __init__(self) -> None:
        self.documents = None
        self.documents_map = {} # id -> doc
        self.embeddings = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")


    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings


    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Text must not be empty")
        return self.model.encode([text])[0]


    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) ==  len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)


    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for doc_embedding, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((_similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        result = []
        for sc, doc in similarities[:limit]:
            result.append({'score':sc,
                        'title': doc['title'],
                        'description':doc['description']
                        })
        return result


class ChunkSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata = None
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']:doc for doc in documents}
        all_chunks = [] # List[str]
        chunk_metadata = [] #List[dict]

        for midx, doc in enumerate(documents):
            if doc['description'].strip()=='':
                continue
            _chunks = semantic_chunking(doc['description'], overlap=1, max_chunk_size=4)
            all_chunks += _chunks
            for cidx in range(len(_chunks)):
                chunk_metadata.append({"movie_idx": midx,
                                       "chunk_idx": cidx,
                                       "total_chunks": len(_chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']:doc for doc in documents}
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0)
        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata['chunks'][idx]
            midx, cidx = metadata['movie_idx'], metadata['chunk_idx']
            sim = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({
                "movie_idx": midx,
                "chunk_idx": cidx,
                "score": sim
                })
            movie_scores[midx] = max(movie_scores[midx], sim)
        movie_scores_sorted = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for midx, score in movie_scores_sorted[:limit]:
            print(midx)
            doc = self.document_map[midx]
            result.append({'id': doc['id'],
                           'title': doc['title'],
                           'document': doc['description'][:100],
                           'score': round(score, SCORE_PRECISION),
                           })
        return result


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_chunks():
    movies = load_movies()
    css = ChunkSemanticSearch()
    embedding = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embedding)} chunked embeddings")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def fixed_sized_chunking(text, overlap, chunk_size=200):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def semantic_chunking(text, overlap=0, max_chunk_size=4):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


def chunk_text_semantic(text, overlap=0, max_chunk_size=4):
    chunks = semantic_chunking(text, overlap, max_chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}")


def chunk_text(text, overlap, chunk_size=200):
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}")


def search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_results = ss.search(query, limit)
    for idx, res in enumerate(search_results):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res['description'][:100])


def search_chunked(query, limit=5):
    css = ChunkSemanticSearch()
    movies = load_movies()
    _ = css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"\n{i+1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def verify_embeddings() -> None:
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
