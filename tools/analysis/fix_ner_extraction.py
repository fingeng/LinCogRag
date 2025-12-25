"""
改进NER实体提取，提高召回率
"""

import re
from typing import List, Set

class EnhancedNERExtractor:
    """增强的NER提取器，提高短文本和常见术语的识别率"""
    
    def __init__(self):
        # 添加医学/生物学常见术语词典
        self.medical_terms = {
            # 细胞和组织
            'cells', 'nucleus', 'lymphocyte', 'monocyte', 'blood', 
            'tissue', 'organ', 'gland', 'membrane',
            
            # 生理过程
            'cardiac cycle', 'systole', 'diastole', 'stroke volume',
            'translation', 'transcription', 'glycolysis', 'metabolism',
            
            # 分子和化合物
            'ATP', 'DNA', 'RNA', 'protein', 'enzyme', 'glucose',
            
            # 解剖结构
            'bone', 'finger', 'hand', 'heart', 'vessel',
            
            # 病理和症状
            'pain', 'disease', 'syndrome', 'disorder',
            
            # 遗传学
            'gene', 'chromosome', 'monosomy', 'trisomy', 'exon', 'intron',
        }
        
        # 医学缩写词
        self.abbreviations = {
            'ATP', 'DNA', 'RNA', 'NHS', 'ECG', 'MRI', 'CT',
        }
    
    def extract_entities_from_question(self, question: str) -> List[str]:
        """从问题中提取实体，包括后备策略"""
        entities = []
        question_lower = question.lower()
        
        # 1. 提取医学术语
        for term in self.medical_terms:
            if term.lower() in question_lower:
                entities.append(term)
        
        # 2. 提取缩写词（保持大写）
        for abbr in self.abbreviations:
            if abbr in question:
                entities.append(abbr)
        
        # 3. 提取专有名词（大写开头的词）
        words = question.split()
        for word in words:
            # 移除标点符号
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                if clean_word not in ['The', 'What', 'Which', 'When', 'Where', 'Who', 'How']:
                    entities.append(clean_word)
        
        # 4. 提取数字+单位的模式
        number_patterns = re.findall(r'\d+[\s-]?(?:mg|ml|kg|mm|cm|m|g|l)', question_lower)
        entities.extend(number_patterns)
        
        return list(set(entities))  # 去重

# 测试
if __name__ == "__main__":
    extractor = EnhancedNERExtractor()
    
    test_questions = [
        "Which cells in the blood do not have a nucleus?",
        "The enzymes of glycolysis are located in the:",
        "The most rapid method to resynthesize ATP during exercise is through:",
        "Which of the following is an example of monosomy?",
    ]
    
    for q in test_questions:
        entities = extractor.extract_entities_from_question(q)
        print(f"\nQuestion: {q}")
        print(f"Entities: {entities}")
