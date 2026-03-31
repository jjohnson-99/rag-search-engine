#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
        embed_text,
        embed_query_text,
        search,
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
    search_parser.add_argument("query", type=str, help="User query to search base on")
    search_parser.add_argument("--limit", type=int, help="Number results to return")


    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text(args.query)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
