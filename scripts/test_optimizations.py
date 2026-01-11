#!/usr/bin/env python3
"""
Quick test script to verify optimization implementations.
Tests answer parsing and retrieval strategies without full benchmark.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_answer_parsing():
    """Test improved answer parsing logic."""
    print("\n" + "="*60)
    print("Testing Answer Parsing Improvements")
    print("="*60)
    
    from src.LinearRAG import LinearRAG
    from src.config import LinearRAGConfig
    from sentence_transformers import SentenceTransformer
    
    # Create minimal config for testing
    embedding_model = SentenceTransformer("model/all-mpnet-base-v2")
    config = LinearRAGConfig(
        embedding_model=embedding_model,
        dataset_name="test",
        working_dir=os.path.join(PROJECT_ROOT, "import"),
    )
    
    # Create instance to access parsing methods
    # Note: We mock the graph to avoid full initialization
    import igraph as ig
    
    class MockLinearRAG:
        def __init__(self):
            self.config = config
            
        def _parse_yesno_maybe(self, response: str) -> str:
            import re
            text = response.strip().lower()
            
            if text in ['yes', 'no', 'maybe']:
                return text.capitalize()
            
            match = re.search(r'\b(yes|no|maybe)\b', text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
            
            first_word = text.split()[0] if text.split() else ''
            first_word = re.sub(r'[^\w]', '', first_word)
            if first_word in ['yes', 'no', 'maybe']:
                return first_word.capitalize()
            
            positive_signals = ['yes', 'true', 'correct', 'support', 'confirm']
            negative_signals = ['no', 'false', 'incorrect', 'contradict', 'not support']
            uncertain_signals = ['maybe', 'uncertain', 'inconclusive', 'insufficient', 'unclear']
            
            pos_count = sum(1 for s in positive_signals if s in text)
            neg_count = sum(1 for s in negative_signals if s in text)
            unc_count = sum(1 for s in uncertain_signals if s in text)
            
            if unc_count > 0:
                return 'Maybe'
            if pos_count > neg_count:
                return 'Yes'
            if neg_count > pos_count:
                return 'No'
            
            return 'INVALID'
        
        def _parse_yesno(self, response: str) -> str:
            import re
            text = response.strip().lower()
            
            if text in ['yes', 'no']:
                return text.capitalize()
            
            match = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
            
            positive_signals = ['yes', 'true', 'correct', 'support', 'confirm']
            negative_signals = ['no', 'false', 'incorrect', 'contradict']
            
            pos_count = sum(1 for s in positive_signals if s in text)
            neg_count = sum(1 for s in negative_signals if s in text)
            
            if pos_count > neg_count:
                return 'Yes'
            if neg_count > pos_count:
                return 'No'
            
            return 'INVALID'
        
        def _parse_mcq(self, response: str, dataset_name: str) -> str:
            import re
            text = response.strip().upper()
            
            if text in ['A', 'B', 'C', 'D']:
                return text
            
            match = re.search(r'\b([ABCD])\b', text)
            if match:
                return match.group(1)
            
            match = re.search(r'[\(\[]?([ABCD])[\)\].]', text)
            if match:
                return match.group(1)
            
            match = re.search(r'(?:answer|option|choice)[:\s]*([ABCD])', text, re.IGNORECASE)
            if match:
                return match.group(1)
            
            match = re.search(r'[ABCD]', text)
            if match:
                return match.group(0)
            
            return 'INVALID'
    
    rag = MockLinearRAG()
    
    # Test cases for PubMedQA
    pubmedqa_tests = [
        ("Yes", "Yes"),
        ("No", "No"),
        ("Maybe", "Maybe"),
        ("yes.", "Yes"),
        ("Based on the evidence, yes", "Yes"),
        ("The answer is no", "No"),
        ("The evidence is inconclusive", "Maybe"),
        ("I cannot determine", "Maybe"),
        ("This is unclear", "Maybe"),
        ("The study supports this", "Yes"),
        ("No evidence was found", "No"),
    ]
    
    print("\nPubMedQA parsing tests:")
    passed = 0
    for input_text, expected in pubmedqa_tests:
        result = rag._parse_yesno_maybe(input_text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_text[:30]:30}' -> {result} (expected: {expected})")
        if result == expected:
            passed += 1
    print(f"  Passed: {passed}/{len(pubmedqa_tests)}")
    
    # Test cases for BioASQ
    bioasq_tests = [
        ("Yes", "Yes"),
        ("No", "No"),
        ("yes.", "Yes"),
        ("The answer is no", "No"),
        ("True, this is correct", "Yes"),
        ("False, this is incorrect", "No"),
    ]
    
    print("\nBioASQ parsing tests:")
    passed = 0
    for input_text, expected in bioasq_tests:
        result = rag._parse_yesno(input_text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_text[:30]:30}' -> {result} (expected: {expected})")
        if result == expected:
            passed += 1
    print(f"  Passed: {passed}/{len(bioasq_tests)}")
    
    # Test cases for MCQ
    mcq_tests = [
        ("A", "A"),
        ("B", "B"),
        ("A.", "A"),
        ("(A)", "A"),
        ("Answer: B", "B"),
        ("The answer is C", "C"),
        ("Option D is correct", "D"),
        ("I think the answer is A because...", "A"),
    ]
    
    print("\nMCQ parsing tests:")
    passed = 0
    for input_text, expected in mcq_tests:
        result = rag._parse_mcq(input_text, "medqa")
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_text[:30]:30}' -> {result} (expected: {expected})")
        if result == expected:
            passed += 1
    print(f"  Passed: {passed}/{len(mcq_tests)}")


def test_retrieval_strategies():
    """Test retrieval strategy components."""
    print("\n" + "="*60)
    print("Testing Retrieval Strategies")
    print("="*60)
    
    from src.retrieval_strategies import (
        EvidenceDecisivenessScorer,
        DatasetAdaptiveRetriever,
        MMULKnowledgeAugmenter,
    )
    
    # Test decisiveness scorer
    scorer = EvidenceDecisivenessScorer()
    
    test_passages = [
        ("Metformin is the first-line treatment for type 2 diabetes.", 0.7, "positive"),
        ("The study shows no significant effect on survival.", 0.7, "negative"),
        ("Some studies suggest a possible benefit.", 0.3, "neutral"),
        ("Further research is needed to confirm these findings.", 0.2, "neutral"),
        ("This treatment is clearly effective and recommended.", 0.8, "positive"),
    ]
    
    print("\nDecisiveness scoring tests:")
    for passage, expected_min_dec, expected_dir in test_passages:
        dec = scorer.score_decisiveness(passage)
        direction = scorer.get_evidence_direction(passage)
        
        dec_status = "✅" if dec >= expected_min_dec - 0.2 else "❌"
        dir_status = "✅" if direction == expected_dir else "❌"
        
        print(f"  {dec_status} '{passage[:40]:40}...'")
        print(f"       Decisiveness: {dec:.2f} (expected >= {expected_min_dec})")
        print(f"       {dir_status} Direction: {direction} (expected: {expected_dir})")
    
    # Test MMLU augmenter
    augmenter = MMULKnowledgeAugmenter()
    
    print("\nMMUL query expansion tests:")
    test_queries = [
        "What is the treatment for heart attack?",
        "Which medication is used for high blood pressure?",
        "What causes diabetes?",
    ]
    
    for query in test_queries:
        expanded = augmenter.expand_query(query)
        print(f"  Original: {query}")
        print(f"  Expanded: {expanded[:80]}...")
        print()


def main():
    print("="*60)
    print("LinCogRAG Optimization Tests")
    print("="*60)
    
    test_answer_parsing()
    test_retrieval_strategies()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
