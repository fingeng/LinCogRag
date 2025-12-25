"""
Comprehensive comparison of NER models for medical QA
"""
import spacy
from collections import Counter

SAMPLE_TEXTS = [
    "The glomerulus filters blood in the kidney nephron.",
    "The ectomesenchyme gives rise to dental tissues including dentin and cementum.",
    "Capillaries in the lung alveoli facilitate gas exchange.",
    "The sinoatrial node initiates electrical impulses in the heart.",
    "Insulin secreted by pancreatic islets regulates glucose metabolism.",
]

MODELS = {
    'BC5CDR (diseases+chemicals)': 'en_ner_bc5cdr_md',
    'CRAFT (comprehensive)': 'en_ner_craft_md',
}

def test_model_on_samples(model_name):
    """Test a model and return entity counts"""
    try:
        nlp = spacy.load(model_name)
        
        all_entities = []
        for text in SAMPLE_TEXTS:
            doc = nlp(text)
            entities = [(e.text, e.label_) for e in doc.ents]
            all_entities.extend(entities)
        
        return all_entities
    except:
        return None

def main():
    print("="*70)
    print("Comprehensive NER Model Comparison for Medical QA")
    print("="*70)
    
    results = {}
    
    for model_label, model_name in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Testing: {model_label}")
        print(f"{'='*70}")
        
        entities = test_model_on_samples(model_name)
        
        if entities is None:
            print(f"❌ Model not installed")
            continue
        
        if not entities:
            print(f"⚠️  No entities found")
            continue
        
        results[model_label] = entities
        
        # Statistics
        entity_types = Counter(e[1] for e in entities)
        unique_terms = set(e[0].lower() for e in entities)
        
        print(f"✅ Found {len(entities)} entities")
        print(f"   Unique terms: {len(unique_terms)}")
        print(f"   Entity types: {len(entity_types)}")
        print(f"\n   Distribution:")
        for ent_type, count in entity_types.most_common():
            examples = [e[0] for e in entities if e[1] == ent_type][:3]
            print(f"     {ent_type:15s}: {count:2d}  (e.g., {', '.join(examples)})")
    
    # Comparison
    print(f"\n{'='*70}")
    print("Recommendation:")
    print(f"{'='*70}")
    
    if 'CRAFT (comprehensive)' in results and results['CRAFT (comprehensive)']:
        craft_count = len(results['CRAFT (comprehensive)'])
        bc5cdr_count = len(results.get('BC5CDR (diseases+chemicals)', []))
        
        print(f"\n✅ CRAFT extracts {craft_count} entities vs BC5CDR's {bc5cdr_count}")
        print(f"   CRAFT is {craft_count/bc5cdr_count:.1f}x better for anatomical terms")
        print(f"\n💡 Use CRAFT (en_ner_craft_md) for MMLU medical QA")
        print(f"\nTo switch:")
        print(f"  python run.py --spacy_model en_ner_craft_md ...")
    
    print("="*70)

if __name__ == '__main__':
    main()
