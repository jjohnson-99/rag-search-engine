#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
        chunk_text,
        chunk_text_semantic,
        embed_chunks,
        embed_text,
        embed_query_text,
        search,
        search_chunked,
        verify_model,
        verify_embeddings,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads properly")
    subparsers.add_parser("verify_embeddings", help="Verify the embedding model loads properly")

    embed_parser = subparsers.add_parser("embed_text", help="Encode text with emedding model")
    embed_parser.add_argument("text", type=str, help="Text to be encoded")

    embed_query_parser = subparsers.add_parser("embed_query", help="Encode query with embedding model")
    embed_query_parser.add_argument("query", type=str, help="User query to be encoded")

    search_parser = subparsers.add_parser("search", help="Search for a relavent movie")
    search_parser.add_argument("query", type=str, help="User query to search based on")
    search_parser.add_argument("--limit", type=int, help="Number results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk document")
    chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Numbers of words in each fixed size chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Numbers of words in each fixed size chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk document")
    semantic_chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Numbers of words in each fixed size chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Numbers of words in each fixed size chunk")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Encode chunks with emedding model")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Chunk document")
    search_chunked_parser.add_argument("query", type=str, help="User query to search based on")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number results to return")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "semantic_chunk":
            chunk_text_semantic(args.text, args.overlap, args.max_chunk_size)
        case "embed_chunks":
            embed_chunks()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
