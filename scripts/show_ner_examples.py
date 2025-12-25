"""
Show detailed NER extraction examples on MMLU questions
"""
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

def load_questions(predictions_path):
    """Load questions from predictions.json"""
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    return [item['question'] for item in data if 'question' in item]

def show_ner_examples(model_path="models/biomedical-ner-all", num_examples=10):
    """Show detailed NER extraction examples"""
    print("="*100)
    print("Detailed NER Extraction Examples")
    print("="*100)
    
    # Load model
    print(f"\n📥 Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device
    )
    
    print(f"✅ Model loaded\n")
    
    # Load questions
    predictions_path = "results/pubmed_mirage_mmlu/predictions.json"
    questions = load_questions(predictions_path)
    
    # Select diverse questions with medical terms
    # 选择包含关键医学术语的问题
    target_questions = []
    keywords = [
        'kidney', 'nephron', 'glomerulus', 'capillary',
        'nerve', 'spinal', 'vertebra', 'atrium', 
        'ventricle', 'blood', 'heart', 'cell',
        'mandible', 'temporomandibular', 'larynx',
        'palatine', 'sinoatrial', 'esophageal'
    ]
    
    for question in questions:
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in keywords):
            target_questions.append(question)
            if len(target_questions) >= num_examples:
                break
    
    # If not enough, add more questions
    if len(target_questions) < num_examples:
        for question in questions:
            if question not in target_questions:
                target_questions.append(question)
            if len(target_questions) >= num_examples:
                break
    
    # Show examples
    print("="*100)
    print(f"Showing {len(target_questions)} Detailed Examples")
    print("="*100)
    
    for i, question in enumerate(target_questions, 1):
        print(f"\n{'='*100}")
        print(f"Example {i}/{len(target_questions)}")
        print(f"{'='*100}")
        
        # Show original question
        print(f"\n📝 Original Question:")
        print(f"   {question}")
        print(f"\n   Length: {len(question)} characters")
        print(f"   Words: {len(question.split())} words")
        
        # Extract entities
        try:
            results = ner(question)
            
            if results:
                print(f"\n✅ Extracted {len(results)} Entities:")
                print(f"\n   {'Entity Text':<40} {'Type':<30} {'Confidence':<10}")
                print(f"   {'-'*40} {'-'*30} {'-'*10}")
                
                for entity in results:
                    entity_text = entity['word'].strip()
                    entity_type = entity['entity_group']
                    score = entity['score']
                    
                    # Highlight in different colors based on type
                    print(f"   {entity_text:<40} {entity_type:<30} {score:>7.2%}")
                
                # Group by entity type
                print(f"\n📊 Entities by Type:")
                from collections import defaultdict
                by_type = defaultdict(list)
                for entity in results:
                    by_type[entity['entity_group']].append(entity['word'].strip())
                
                for ent_type, entities in sorted(by_type.items()):
                    print(f"   • {ent_type}: {', '.join(entities)}")
                
                # Check for missing medical terms
                print(f"\n🔍 Medical Term Analysis:")
                found_keywords = []
                missing_keywords = []
                
                extracted_texts = [e['word'].strip().lower() for e in results]
                question_lower = question.lower()
                
                for keyword in keywords:
                    if keyword in question_lower:
                        if any(keyword in text for text in extracted_texts):
                            found_keywords.append(keyword)
                            print(f"   ✅ '{keyword}' - FOUND in entities")
                        else:
                            missing_keywords.append(keyword)
                            print(f"   ❌ '{keyword}' - MISSING (present in question)")
                
                if not found_keywords and not missing_keywords:
                    print(f"   ℹ️  No target medical keywords in this question")
                
            else:
                print(f"\n⚠️  No entities extracted")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print(f"\n{'='*100}")
    
    # Summary statistics
    print(f"\n{'='*100}")
    print("Summary Statistics")
    print(f"{'='*100}")
    
    total_entities = 0
    total_questions_with_entities = 0
    all_entity_types = set()
    
    for question in target_questions:
        try:
            results = ner(question)
            if results:
                total_questions_with_entities += 1
                total_entities += len(results)
                for entity in results:
                    all_entity_types.add(entity['entity_group'])
        except:
            pass
    
    print(f"\nQuestions analyzed: {len(target_questions)}")
    print(f"Questions with entities: {total_questions_with_entities} ({total_questions_with_entities/len(target_questions)*100:.1f}%)")
    print(f"Total entities extracted: {total_entities}")
    print(f"Average entities per question: {total_entities/len(target_questions):.2f}")
    print(f"Unique entity types: {len(all_entity_types)}")
    print(f"\nEntity types: {', '.join(sorted(all_entity_types))}")
    
    print(f"\n{'='*100}")

if __name__ == '__main__':
    show_ner_examples(num_examples=10)
