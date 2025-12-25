# /home/maoxy23/projects/LinearRAG/examples/medqa/run_medqa.py
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import torch

from examples.pubmedqa.build_pubmed_corpus import load_pubmed_corpus
from src.linearrag.core.retriever import LinearRAGRetriever
from src.linearrag.core.generator import Generator
from src.linearrag.core.entity_processor import EntityProcessor


def load_medqa_questions(medqa_path: str) -> List[Dict[str, Any]]:
    """Load MedQA questions"""
    with open(medqa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        question_text = item["question"]
        options = item["options"]
        answer_idx = item["answer_idx"]
        
        # Format: question + options
        formatted_question = f"{question_text}\n"
        for key, value in options.items():
            formatted_question += f"{key}. {value}\n"
        
        questions.append({
            "question": formatted_question,
            "answer": options[answer_idx],
            "answer_idx": answer_idx,
            "options": options,
            "metamap_phrases": item.get("metamap_phrases", [])
        })
    
    return questions


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    medqa_path = project_root / "data" / "medqa" / "questions" / "US" / "train.jsonl.metamap"
    pubmed_dir = project_root / "data" / "pubmed"
    cache_dir = project_root / "cache" / "medqa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete old NER cache
    ner_cache_file = cache_dir / "entity_cache.json"
    if ner_cache_file.exists():
        print(f"🗑️  Removing old NER cache: {ner_cache_file}")
        ner_cache_file.unlink()
    
    # Load questions
    print("Loading MedQA questions...")
    questions = load_medqa_questions(str(medqa_path))
    print(f"✅ Loaded {len(questions)} questions")
    
    # Load PubMed passages
    print("Loading PubMed passages (limit: 1000)...")
    passages = load_pubmed_corpus(str(pubmed_dir), max_passages=1000)
    print(f"✅ Loaded {len(passages)} passages")
    
    # Initialize components
    print("\n" + "="*50)
    print("Initializing LinearRAG components...")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize entity processor with cache
    entity_processor = EntityProcessor(
        cache_file=str(ner_cache_file),
        device=device
    )
    
    # Initialize retriever
    retriever = LinearRAGRetriever(
        entity_processor=entity_processor,
        device=device
    )
    
    # Build corpus
    print("\nBuilding corpus from passages...")
    retriever.build_corpus(passages)
    
    # Initialize generator
    generator = Generator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device=device
    )
    
    # Process questions
    print("\n" + "="*50)
    print("Processing questions...")
    print("="*50)
    
    results = []
    correct = 0
    
    for i, item in enumerate(tqdm(questions[:10], desc="Evaluating")):  # Test on first 10
        question = item["question"]
        
        # Retrieve relevant passages
        retrieved = retriever.retrieve(question, top_k=3)
        context = "\n\n".join([doc["content"] for doc in retrieved])
        
        # Generate answer
        prompt = f"""Based on the following context, answer the multiple choice question.

Context:
{context}

Question:
{question}

Please provide only the letter of the correct answer (A, B, C, or D)."""
        
        answer = generator.generate(prompt, max_length=10).strip()
        
        # Extract first letter
        predicted_idx = None
        for char in answer:
            if char.upper() in item["options"]:
                predicted_idx = char.upper()
                break
        
        is_correct = (predicted_idx == item["answer_idx"])
        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "predicted": predicted_idx,
            "correct": item["answer_idx"],
            "is_correct": is_correct,
            "retrieved_docs": len(retrieved)
        })
        
        if (i + 1) % 5 == 0:
            accuracy = correct / (i + 1)
            print(f"\nProgress: {i+1}/10, Accuracy: {accuracy:.2%}")
    
    # Final results
    final_accuracy = correct / len(results)
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {final_accuracy:.2%}")
    
    # Save results
    output_file = cache_dir / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": final_accuracy,
            "total": len(results),
            "correct": correct,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()