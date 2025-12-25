#!/usr/bin/env python3
"""
快速优化脚本 - 无需修改核心代码的参数调优
用于快速测试不同超参数配置的效果
"""

import json
import time
from datetime import datetime

# 优化配置方案
OPTIMIZATION_CONFIGS = {
    "baseline": {
        "name": "基线配置 (当前)",
        "max_iterations": 3,
        "iteration_threshold": 0.1,
        "top_k_sentence": 3,
        "retrieval_top_k": 32,
        "expected_speed": "150s/问题",
        "description": "当前使用的配置，速度最慢但可能效果最好"
    },
    
    "quick_fix": {
        "name": "快速优化 (推荐)",
        "max_iterations": 2,
        "iteration_threshold": 0.3,
        "top_k_sentence": 5,
        "retrieval_top_k": 32,
        "expected_speed": "30-50s/问题",
        "description": "提高阈值+减少迭代，预计提速3-5倍"
    },
    
    "aggressive": {
        "name": "激进优化",
        "max_iterations": 1,
        "iteration_threshold": 0.5,
        "top_k_sentence": 10,
        "retrieval_top_k": 32,
        "expected_speed": "10-20s/问题",
        "description": "只做1次迭代，速度最快但可能损失效果"
    },
    
    "balanced": {
        "name": "平衡配置",
        "max_iterations": 2,
        "iteration_threshold": 0.25,
        "top_k_sentence": 8,
        "retrieval_top_k": 32,
        "expected_speed": "20-40s/问题",
        "description": "速度和效果的平衡点"
    }
}


def print_optimization_guide():
    """打印优化指南"""
    print("=" * 80)
    print("LinearRAG 医疗领域优化指南")
    print("=" * 80)
    print("\n📊 当前性能分析:")
    print("   - 检索速度: 60-150秒/问题 (严重过慢)")
    print("   - 预计总时间: 21-53小时 (1273个问题)")
    print("   - 瓶颈: 图规模过大 (21万实体) + 迭代计算昂贵")
    
    print("\n" + "=" * 80)
    print("优化配置方案对比")
    print("=" * 80)
    
    for config_name, config in OPTIMIZATION_CONFIGS.items():
        print(f"\n【{config['name']}】")
        print(f"   配置名: {config_name}")
        print(f"   max_iterations: {config['max_iterations']}")
        print(f"   iteration_threshold: {config['iteration_threshold']}")
        print(f"   top_k_sentence: {config['top_k_sentence']}")
        print(f"   预期速度: {config['expected_speed']}")
        print(f"   说明: {config['description']}")
    
    print("\n" + "=" * 80)
    print("快速操作指南")
    print("=" * 80)
    print("\n1️⃣ 修改配置文件 (src/config.py):")
    print("   找到 LinearRAGConfig 类，修改默认参数")
    print("\n2️⃣ 停止当前运行:")
    print("   kill 3478849  # 或者 pkill -f 'run.py'")
    print("\n3️⃣ 使用优化配置重新运行:")
    print("   python run.py \\")
    print("       --use_hf_ner \\")
    print("       --embedding_model model/all-mpnet-base-v2 \\")
    print("       --dataset_name pubmed \\")
    print("       --llm_model gpt-4o-mini \\")
    print("       --max_workers 8 \\")
    print("       --use_mirage \\")
    print("       --mirage_dataset medqa \\")
    print("       --chunks_limit 10000 \\")
    print("       > medqa_optimized.log 2>&1 &")
    print("\n4️⃣ 监控性能:")
    print("   tail -f medqa_optimized.log | grep 'Retrieving:'")
    
    print("\n" + "=" * 80)
    print("⚠️ 重要提示")
    print("=" * 80)
    print("1. 建议先用 'quick_fix' 配置测试100个问题")
    print("2. 对比速度和准确率后，再决定是否调整")
    print("3. 如果准确率下降<2%，速度提升>3x，就值得采用")
    print("4. 可以用小数据集 (--questions_limit 100) 快速验证")


def generate_config_file(config_name="quick_fix"):
    """生成优化后的配置文件"""
    config = OPTIMIZATION_CONFIGS[config_name]
    
    config_content = f'''from dataclasses import dataclass
"""
注意：该文件仅用于展示/记录一次实验配置，不依赖具体 LLM wrapper。
主流程请使用 `run.py` + `src/pipeline.py`。
"""

@dataclass
class LinearRAGConfig:
    """
    LinearRAG配置 - {config['name']}
    生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    预期速度: {config['expected_speed']}
    """
    def __init__(
        self,
        embedding_model,
        dataset_name,
        spacy_model="en_ner_bc5cdr_md",
        max_workers=8,
        llm_model=None,
        use_hf_ner=True,
        use_enhanced_ner=True,
        working_dir="import",
        batch_size=32,
        retrieval_top_k={config['retrieval_top_k']},  # ✅ 优化
        max_iterations={config['max_iterations']},  # ✅ 优化: 减少迭代次数
        iteration_threshold={config['iteration_threshold']},  # ✅ 优化: 提高阈值
        top_k_sentence={config['top_k_sentence']},  # ✅ 优化: 增加句子数
        passage_ratio=0.7,
        passage_node_weight=1.0,
        damping=0.85,
    ):
        # Model parameters
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.llm_model = llm_model
        
        # NER strategy
        self.use_hf_ner = use_hf_ner
        self.use_enhanced_ner = use_enhanced_ner
        
        # Dataset parameters
        self.dataset_name = dataset_name
        self.working_dir = working_dir
        
        # Processing parameters
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Retrieval parameters
        self.retrieval_top_k = retrieval_top_k
        self.max_iterations = max_iterations
        self.iteration_threshold = iteration_threshold
        self.top_k_sentence = top_k_sentence
        
        # Graph parameters
        self.passage_ratio = passage_ratio
        self.passage_node_weight = passage_node_weight
        self.damping = damping
'''
    
    output_path = f"src/config_optimized_{config_name}.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n✅ 已生成优化配置文件: {output_path}")
    print(f"   配置: {config['name']}")
    print(f"   预期速度: {config['expected_speed']}")
    
    # 生成使用说明
    usage_content = f"""
# 使用 {config['name']} 配置

## 1. 备份原配置
cp src/config.py src/config_backup.py

## 2. 替换为优化配置
cp {output_path} src/config.py

## 3. 重新运行
kill $(pgrep -f "run.py")
python run.py \\
    --use_hf_ner \\
    --embedding_model model/all-mpnet-base-v2 \\
    --dataset_name pubmed \\
    --llm_model gpt-4o-mini \\
    --max_workers 8 \\
    --use_mirage \\
    --mirage_dataset medqa \\
    --chunks_limit 10000 \\
    --questions_limit 100 \\
    > medqa_{config_name}.log 2>&1 &

## 4. 监控性能
tail -f medqa_{config_name}.log

## 5. 对比结果
# 对比速度: grep "Retrieving:" medqa_*.log
# 对比准确率: 等待运行完成后查看最终准确率
"""
    
    usage_path = f"usage_{config_name}.sh"
    with open(usage_path, 'w', encoding='utf-8') as f:
        f.write(usage_content)
    print(f"✅ 已生成使用说明: {usage_path}")


def analyze_current_log():
    """分析当前日志文件"""
    import re
    
    log_file = "medqa_full_fixed.log"
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取检索进度
        retrieval_pattern = r'Retrieving:\s+(\d+)%.*?(\d+)/(\d+).*?\[([^<]+)<'
        matches = re.findall(retrieval_pattern, content)
        
        if matches:
            last_match = matches[-1]
            percent, current, total, elapsed = last_match
            print("\n📈 当前进度分析:")
            print(f"   已完成: {current}/{total} ({percent}%)")
            print(f"   已用时: {elapsed}")
            
            # 计算平均速度
            if ',' in elapsed:
                parts = elapsed.split(',')
                hours = int(parts[0].split(':')[0])
                minutes = int(parts[0].split(':')[1])
                seconds = int(parts[0].split(':')[2])
                total_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                time_parts = elapsed.split(':')
                if len(time_parts) == 3:
                    total_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                else:
                    total_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
            
            avg_time = total_seconds / int(current)
            remaining = (int(total) - int(current)) * avg_time
            
            print(f"   平均速度: {avg_time:.1f}秒/问题")
            print(f"   预计剩余: {remaining/3600:.1f}小时")
            
            # 提供建议
            if avg_time > 100:
                print("\n⚠️ 速度严重过慢，强烈建议:")
                print("   1. 立即停止当前运行 (kill 3478849)")
                print("   2. 使用 'quick_fix' 配置重新运行")
                print("   3. 预计可提速至 30-50秒/问题")
            elif avg_time > 50:
                print("\n⚠️ 速度较慢，建议优化配置")
            else:
                print("\n✅ 速度尚可")
        
        # 提取图统计信息
        if "Entity embeddings:" in content:
            entity_match = re.search(r'Entity embeddings: \((\d+),', content)
            passage_match = re.search(r'Passage embeddings: \((\d+),', content)
            sentence_match = re.search(r'Sentence embeddings: \((\d+),', content)
            
            if entity_match:
                print("\n📊 图规模统计:")
                print(f"   实体数: {entity_match.group(1)}")
                print(f"   文档数: {passage_match.group(1)}")
                print(f"   句子数: {sentence_match.group(1)}")
                
                entities = int(entity_match.group(1))
                if entities > 200000:
                    print(f"\n⚠️ 实体数过多 ({entities:,})，建议:")
                    print("   1. 过滤低频实体 (出现次数<3)")
                    print("   2. 提高 iteration_threshold 限制扩散范围")
    
    except FileNotFoundError:
        print(f"⚠️ 未找到日志文件: {log_file}")
    except Exception as e:
        print(f"⚠️ 分析日志时出错: {e}")


if __name__ == "__main__":
    print_optimization_guide()
    print("\n" + "=" * 80)
    
    # 分析当前日志
    analyze_current_log()
    
    print("\n" + "=" * 80)
    print("是否生成优化配置文件?")
    print("=" * 80)
    print("\n选择配置方案:")
    for i, (key, config) in enumerate(OPTIMIZATION_CONFIGS.items(), 1):
        print(f"   {i}. {config['name']} ({config['expected_speed']})")
    
    print("\n推荐: 2 (快速优化)")
    print("输入数字生成配置，或按 Enter 跳过:")
    
    choice = input().strip()
    
    if choice:
        config_list = list(OPTIMIZATION_CONFIGS.keys())
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(config_list):
                config_name = config_list[idx]
                generate_config_file(config_name)
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入数字")
    
    print("\n✅ 完成! 请查看生成的文件和说明")
