#!/usr/bin/env python3
"""
性能对比分析脚本
对比原始配置、参数优化、候选集预筛选三个版本的性能
"""

import re
import json
from pathlib import Path

def parse_log_file(log_file):
    """解析日志文件，提取关键信息"""
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取检索速度
    retrieval_times = re.findall(r'(\d+\.\d+)s/it', content)
    if retrieval_times:
        speeds = [float(t) for t in retrieval_times[-10:]]  # 取最后10个
        avg_speed = sum(speeds) / len(speeds)
    else:
        avg_speed = None
    
    # 提取准确率
    accuracy_match = re.search(r'LLM Accuracy:\s+(\d+\.\d+)%', content)
    llm_accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    
    contain_match = re.search(r'Contain Accuracy:\s+(\d+\.\d+)%', content)
    contain_accuracy = float(contain_match.group(1)) if contain_match else None
    
    # 提取完成问题数
    total_match = re.search(r'Total questions:\s+(\d+)', content)
    total_questions = int(total_match.group(1)) if total_match else None
    
    # 检查是否完成
    completed = 'Overall Results' in content
    
    return {
        'avg_speed': avg_speed,
        'llm_accuracy': llm_accuracy,
        'contain_accuracy': contain_accuracy,
        'total_questions': total_questions,
        'completed': completed,
        'log_file': log_file
    }


def main():
    print("=" * 80)
    print("LinearRAG 性能优化对比分析")
    print("=" * 80)
    print()
    
    # 定义要对比的版本
    versions = [
        {
            'name': '原始配置',
            'log_file': 'medqa_full_fixed.log',
            'description': 'max_iterations=3, threshold=0.1, 无候选集过滤'
        },
        {
            'name': '参数优化',
            'log_file': 'medqa_quick_fix_100q.log',
            'description': 'max_iterations=2, threshold=0.3, 无候选集过滤'
        },
        {
            'name': '候选集预筛选',
            'log_file': 'medqa_candidate_filtering_100q.log',
            'description': 'max_iterations=2, threshold=0.3, top-200候选集'
        }
    ]
    
    results = []
    for version in versions:
        data = parse_log_file(version['log_file'])
        if data:
            data.update({
                'name': version['name'],
                'description': version['description']
            })
            results.append(data)
    
    if not results:
        print("⚠️  未找到日志文件，请先运行测试")
        return
    
    # 打印对比表格
    print("📊 性能对比表")
    print("-" * 80)
    print(f"{'版本':<15} {'检索速度':<12} {'LLM准确率':<12} {'Contain准确率':<15} {'状态':<10}")
    print("-" * 80)
    
    baseline_speed = None
    baseline_llm = None
    
    for i, result in enumerate(results):
        name = result['name']
        speed = result['avg_speed']
        llm_acc = result['llm_accuracy']
        contain_acc = result['contain_accuracy']
        completed = '✅ 完成' if result['completed'] else '🔄 运行中'
        
        # 计算相对变化
        if i == 0:
            baseline_speed = speed
            baseline_llm = llm_acc
            speed_str = f"{speed:.1f}s" if speed else "N/A"
            llm_str = f"{llm_acc:.1f}%" if llm_acc else "N/A"
        else:
            if speed and baseline_speed:
                speedup = baseline_speed / speed
                speed_str = f"{speed:.1f}s ({speedup:.1f}x)"
            else:
                speed_str = f"{speed:.1f}s" if speed else "N/A"
            
            if llm_acc and baseline_llm:
                diff = llm_acc - baseline_llm
                llm_str = f"{llm_acc:.1f}% ({diff:+.1f}%)"
            else:
                llm_str = f"{llm_acc:.1f}%" if llm_acc else "N/A"
        
        contain_str = f"{contain_acc:.1f}%" if contain_acc else "N/A"
        
        print(f"{name:<15} {speed_str:<12} {llm_str:<12} {contain_str:<15} {completed:<10}")
    
    print("-" * 80)
    print()
    
    # 打印详细信息
    print("📝 详细配置")
    print("-" * 80)
    for result in results:
        print(f"\n【{result['name']}】")
        print(f"   配置: {result['description']}")
        print(f"   日志: {result['log_file']}")
        if result['avg_speed']:
            print(f"   平均速度: {result['avg_speed']:.2f} 秒/问题")
        if result['llm_accuracy']:
            print(f"   LLM准确率: {result['llm_accuracy']:.2f}%")
        if result['contain_accuracy']:
            print(f"   Contain准确率: {result['contain_accuracy']:.2f}%")
        print(f"   状态: {'完成' if result['completed'] else '运行中'}")
    
    print()
    print("=" * 80)
    
    # 给出建议
    if len(results) >= 2:
        latest = results[-1]
        if latest['completed']:
            if latest['avg_speed']:
                print("\n💡 优化建议:")
                if latest['avg_speed'] < 10:
                    print("   ✅ 速度已优化到位 (< 10秒/问题)")
                    print("   ✅ 可以运行完整测试 (1273个问题)")
                elif latest['avg_speed'] < 20:
                    print("   🎯 速度良好，但还有优化空间")
                    print("   💡 可以考虑进一步优化 (如限制句子数量)")
                else:
                    print("   ⚠️  速度仍需优化")
                    print("   💡 检查候选集过滤是否正确启用")
        else:
            print("\n⏳ 测试运行中，请稍后再次运行此脚本查看结果")
    
    print()


if __name__ == "__main__":
    main()
