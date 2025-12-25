"""
Compare different NER models on sample PubMed text
"""
import spacy
import warnings
from collections import Counter

# 忽略版本警告
warnings.filterwarnings('ignore', category=UserWarning)

# Sample text from your PubMed data
SAMPLE_TEXTS = [
    # Your actual passage
    "(--)-alpha-Bisabolol has a primary antipeptic action depending on dosage, which is not caused by an alteration of the pH-value. The proteolytic activity of pepsin is reduced by 50 percent through addition of bisabolol in the ratio of 1/0.5.",
    
    # Clinical scenario (like MMLU)
    "A 45-year-old patient presents with chest pain and dyspnea. ECG shows ST elevation. Immediate treatment with aspirin and nitroglycerin was initiated.",
    
    # Pharmacology
    "Metformin is a biguanide medication used to treat type 2 diabetes mellitus. It works by decreasing hepatic glucose production and improving insulin sensitivity.",
]

MODELS = {
    'SciSpaCy (current)': 'en_core_sci_scibert',
    'BC5CDR (recommended)': 'en_ner_bc5cdr_md',
    # 'CRAFT': 'en_ner_craft_md',  # Uncomment if installed
}

def analyze_text_with_model(text, model_name):
    """Extract entities using given model"""
    try:
        # 🔧 FIX: Disable problematic components to avoid version conflicts
        disable_components = []
        if 'scibert' in model_name.lower():
            disable_components = ['tagger', 'parser', 'attribute_ruler']  # Only keep NER
        
        nlp = spacy.load(model_name, disable=disable_components)
        doc = nlp(text)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities, None
    except Exception as e:
        return None, str(e)

def main():
    print("="*70)
    print("NER Model Comparison on PubMed Texts")
    print("="*70)
    
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        print(f"\n{'='*70}")
        print(f"Text {i}: {text[:80]}...")
        print(f"{'='*70}\n")
        
        for model_label, model_name in MODELS.items():
            print(f"{model_label} ({model_name}):")
            
            entities, error = analyze_text_with_model(text, model_name)
            
            if error:
                print(f"   ❌ Error: {error[:100]}...")
                print(f"   Install: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{model_name}-0.5.1.tar.gz")
            elif entities is None:
                print(f"   ❌ Model not installed")
                print(f"   Install: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{model_name}-0.5.1.tar.gz")
            elif not entities:
                print(f"   ⚠️  No entities found")
            else:
                print(f"   ✅ Found {len(entities)} entities:")
                
                # Group by type
                by_type = Counter(ent[1] for ent in entities)
                for ent_type, count in by_type.items():
                    examples = [e[0] for e in entities if e[1] == ent_type][:3]
                    print(f"      {ent_type}: {count} ({', '.join(examples)})")
            
            print()
    
    print("="*70)
    print("💡 Recommendation:")
    print("   Use 'en_ner_bc5cdr_md' for:")
    print("   - Better disease recognition")
    print("   - Better drug/chemical recognition")
    print("   - Higher precision for clinical QA")
    print("="*70)

if __name__ == '__main__':
    main()
