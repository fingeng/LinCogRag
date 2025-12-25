#!/usr/bin/env python3
"""
完整性能对比分析 - 包含所有优化版本
"""

import re
from pathlib import Path

def parse_log(log_file):
    """解析日志文件"""
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取速度
    speeds = re.findall(r'(\d+\.\d+)s/it', content)
    avg_speed = sum(float(s) for s in speeds[-20:]) / len(speeds[-20:]) if speeds else None
    
    # 提取准确率
    llm_acc = re.search(r'LLM Accuracy:\s+(\d+\.\d+)%', content)
    contain_acc = re.search(r'Contain Accuracy:\s+(\d+\.\d+)%', content)
    
    return {
        'speed': avg_speed,
        'llm_acc': float(llm_acc.group(1)) if llm_acc else None,
        'contain_acc': float(contain_acc.group(1)) if contain_acc else None,
        'completed': 'Overall Results' in content
    }


def main():
    print("=" * 90)
    print(" " * 25 + "LinearRAG 完整性能对比")
    print("=" * 90)
    print()
    
    versions = [
        ('原始配置', 'medqa_full_fixed.log', 'iter=3, thresh=0.1, 无过滤'),
        ('参数优化', 'medqa_quick_fix_100q.log', 'iter=2, thresh=0.3, 无过滤'),
        ('候选池200', 'medqa_candidate_filtering_100q.log', 'iter=2, thresh=0.3, pool=200'),
        ('候选池500', 'medqa_pool500_accuracy_test.log', 'iter=2, thresh=0.3, pool=500+阈值过滤'),
    ]
    
    results = []
    for name, log_file, config in versions:
        data = parse_log(log_file)
        if data:
            results.append((name, config, data))
    
    if not results:
        print("⚠️  未找到日志文件")
        return
    
    # 打印表格
    print("📊 完整性能对比表")
    print("-" * 90)
    print(f"{'版本':<12} {'配置':<35} {'速度':<12} {'准确率':<10} {'状态':<8}")
    print("-" * 90)
    
    baseline_speed = None
    baseline_acc = None
    
    for i, (name, config, data) in enumerate(results):
        speed = data['speed']
        acc = data['llm_acc']
        status = '✅' if data['completed'] else '🔄'
        
        if i == 0:
            baseline_speed = speed
            baseline_acc = acc
            speed_str = f"{speed:.1f}s" if speed else "N/A"
            acc_str = f"{acc:.1f}%" if acc else "N/A"
        else:
            if speed and baseline_speed:
                speedup = baseline_speed / speed
                speed_str = f"{speed:.1f}s ({speedup:.1f}x)"
            else:
                speed_str = f"{speed:.1f}s" if speed else "N/A"
            
            if acc and baseline_acc:
                diff = acc - baseline_acc
                acc_str = f"{acc:.1f}% ({diff:+.1f})"
            else:
                acc_str = f"{acc:.1f}%" if acc else "N/A"
        
        print(f"{name:<12} {config:<35} {speed_str:<12} {acc_str:<10} {status:<8}")
    
    print("-" * 90)
    print()
    
    # 分析最新版本
    if results:
        latest_name, latest_config, latest_data = results[-1]
        
        if latest_data['completed']:
            print("📈 最新版本分析")
            print("-" * 90)
            print(f"版本: {latest_name}")
            print(f"配置: {latest_config}")
            print(f"速度: {latest_data['speed']:.2f} 秒/问题")
            print(f"LLM准确率: {latest_data['llm_acc']:.1f}%")
            print(f"Contain准确率: {latest_data['contain_acc']:.1f}%")
            print()
            
            # 给出建议
            if latest_data['llm_acc']:
                print("💡 优化建议:")
                
                if latest_data['speed'] < 5 and latest_data['llm_acc'] >= 72:
                    print("   ✅ 速度和准确率都达到优秀水平!")
                    print("   ✅ 建议运行完整测试 (1273个问题)")
                    print(f"   ⏱️  预计完成时间: {latest_data['speed'] * 1273 / 3600:.1f} 小时")
                    
                elif latest_data['speed'] < 5 and latest_data['llm_acc'] < 72:
                    print(f"   ⚠️  准确率 ({latest_data['llm_acc']:.1f}%) 略低于目标 (72%)")
                    print("   💡 建议:")
                    print("      - 进一步扩大候选池到800")
                    print("      - 或降低句子相似度阈值到0.2")
                    
                elif latest_data['speed'] >= 5 and latest_data['llm_acc'] >= 72:
                    print("   ✅ 准确率优秀!")
                    print(f"   💡 速度 ({latest_data['speed']:.1f}秒) 可以接受")
                    print("   ✅ 可以运行完整测试")
                    
                else:
                    print("   💡 继续优化中...")
        else:
            print("⏳ 测试运行中，约5-10分钟完成...")
            print()
    
    print()
    print("=" * 90)
    
    # 性能提升总结
    if len(results) >= 2:
        first_speed = results[0][2]['speed']
        last_speed = results[-1][2]['speed']
        first_acc = results[0][2]['llm_acc']
        last_acc = results[-1][2]['llm_acc']
        
        if all([first_speed, last_speed, first_acc, last_acc]):
            print()
            print("🎯 优化成果总结")
            print("-" * 90)
            print(f"速度优化: {first_speed:.1f}秒 → {last_speed:.1f}秒 (提速 {first_speed/last_speed:.1f}x)")
            print(f"准确率: {first_acc:.1f}% → {last_acc:.1f}% (变化 {last_acc-first_acc:+.1f}%)")
            
            # 完成1273问题的时间对比
            time_before = first_speed * 1273 / 3600
            time_after = last_speed * 1273 / 3600
            print(f"\n完成1273问题:")
            print(f"  优化前: {time_before:.1f} 小时")
            print(f"  优化后: {time_after:.1f} 小时")
            print(f"  节省: {time_before - time_after:.1f} 小时 ({(1-time_after/time_before)*100:.0f}%)")
            print()


if __name__ == "__main__":
    main()
