import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LinearRAG + MIRAGE/Med benchmarks runner")

    # Model parameters
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer embedding model name/path",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_ner_bc5cdr_md",
        help="spaCy/scispaCy model name for NER (e.g., en_ner_bc5cdr_md)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (OpenAI compatible)",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pubmed",
        help="Corpus name (currently used for output namespace; passages default to dataset/pubmed/chunk)",
    )
    parser.add_argument(
        "--use_mirage",
        action="store_true",
        help="Use local MIRAGE rawdata loaders (medqa/medmcqa/mmlu/pubmedqa/bioasq)",
    )
    parser.add_argument(
        "--mirage_dataset",
        type=str,
        nargs="+",
        default=["medqa"],
        choices=["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"],
        help="MIRAGE dataset(s) to evaluate",
    )

    # Processing parameters
    parser.add_argument("--max_workers", type=int, default=4, help="Max workers for NER processing")
    parser.add_argument("--chunks_limit", type=int, default=None, help="Limit number of passages loaded")
    parser.add_argument("--questions_limit", type=int, default=None, help="Limit number of questions per dataset")

    # NER strategy
    parser.add_argument("--use_hf_ner", action="store_true", help="Enable HF NER supplement")
    parser.add_argument("--use_enhanced_ner", action="store_true", help="Enable enhanced hybrid NER pipeline")

    # Output / run controls
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory to save logs/results (default: artifacts/)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name (default: auto timestamped name)",
    )
    parser.add_argument(
        "--reuse_index",
        type=str,
        default=None,
        help="Reuse an existing index directory under import/ (e.g. pubmed_mirage_medqa). If set, bypasses auto naming.",
    )

    return parser


def parse_args():
    return build_arg_parser().parse_args()


