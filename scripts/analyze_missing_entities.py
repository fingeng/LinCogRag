"""
Analyze questions that failed entity extraction
"""
import re

def analyze_log(log_file="job3.log"):
    """Analyze missing entity warnings"""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract all warning messages
    pattern = r"⚠️  No entities extracted from question: '(.*?)\.\.\.'"
    matches = re.findall(pattern, content)
    
    print("="*70)
    print(f"Questions Missing Entity Extraction")
    print("="*70)
    print(f"\nTotal questions missing entities: {len(matches)}")
    print(f"Percentage: {len(matches)/1089*100:.2f}%")
    
    print(f"\n{'='*70}")
    print("Sample Questions (First 10):")
    print(f"{'='*70}")
    
    for i, q in enumerate(matches[:10], 1):
        print(f"\n{i}. {q[:70]}...")
    
    # Categorize by topic
    print(f"\n{'='*70}")
    print("Topic Analysis:")
    print(f"{'='*70}")
    
    medical_keywords = ['patient', 'clinical', 'medical', 'hospital', 'catheter', 
                       'drug', 'infection', 'breathing', 'pulse', 'urinary']
    general_keywords = ['hardy', 'weinberg', 'rational', 'choice', 'evolution', 
                       'intracellular', 'successional']
    
    medical_count = sum(1 for q in matches if any(k in q.lower() for k in medical_keywords))
    general_count = sum(1 for q in matches if any(k in q.lower() for k in general_keywords))
    
    print(f"\nMedical/Clinical questions: {medical_count} ({medical_count/len(matches)*100:.1f}%)")
    print(f"General biology questions: {general_count} ({general_count/len(matches)*100:.1f}%)")
    print(f"Other: {len(matches)-medical_count-general_count}")
    
    print(f"\n{'='*70}")
    print("Impact Assessment:")
    print(f"{'='*70}")
    
    extraction_rate = (1089 - len(matches)) / 1089 * 100
    print(f"\nEntity extraction success rate: {extraction_rate:.2f}%")
    
    if extraction_rate > 98:
        print("✅ EXCELLENT - Extraction rate > 98%")
    elif extraction_rate > 95:
        print("✅ GOOD - Extraction rate > 95%")
    elif extraction_rate > 90:
        print("⚠️  ACCEPTABLE - Extraction rate > 90%")
    else:
        print("❌ POOR - Extraction rate < 90%")
    
    print(f"\n💡 Recommendation:")
    if medical_count > len(matches) * 0.5:
        print("   → These are medical questions - enhance medical keyword dictionary")
    else:
        print("   → These are general questions - system will use dense retrieval fallback")
        print("   → This is acceptable as the graph retrieval may not help for non-domain questions")

if __name__ == '__main__':
    analyze_log()
