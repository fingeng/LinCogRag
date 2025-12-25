"""
Verify that MMLU questions include options.

该脚本使用统一的 MIRAGE 本地 loader：`src.dataset_loader.load_mirage_questions_local`，
避免依赖旧的 `src.mirage_loader`（已弃用/将删除）。
"""
import sys
import os

# 🔧 FIX: Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset_loader import load_mirage_questions_local

def verify_format():
    """Verify MMLU question format"""
    print("="*70)
    print("Verifying MMLU Question Format")
    print("="*70)
    
    questions = load_mirage_questions_local(["mmlu"])
    
    print(f"\n✅ Loaded {len(questions)} questions\n")
    
    # Check first 3 questions
    print("Sample questions:")
    print("="*70)
    
    for i, q in enumerate(questions[:3], 1):
        print(f"\nQuestion {i}:")
        print(f"{q['question']}")
        print(f"\nGold Answer: {q['answer']}")
        print(f"{'-'*70}")
    
    # Check if options are included
    has_options = all('Options:' in q['question'] or 
                     any(opt in q['question'] for opt in ['A.', 'B.', 'C.', 'D.'])
                     for q in questions[:10])
    
    if has_options:
        print("\n✅ Questions include options (ABCD)")
    else:
        print("\n❌ Questions missing options!")
        print("   Fix: Update src/mirage_loader.py to include options")
    
    return has_options

if __name__ == '__main__':
    verify_format()
