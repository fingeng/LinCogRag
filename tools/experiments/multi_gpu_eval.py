import os
import sys
import json
import torch
import torch.multiprocessing as mp
from pathlib import Path
import logging
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from linear_rag import LinearRAG, LinearRAGConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 数据集配置
DATASETS = {
    0: "hotpotqa",
    1: "2wikimultihopqa", 
    2: "musique",
    3: "bamboogle",
    4: "squad"
}

def build_graph_once(rank=0):
    """在一个GPU上构建图（只执行一次）"""
    try:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        
        logging.info(f"[GPU {rank}] Building graph on {device}...")
        
        config = LinearRAGConfig(
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            top_p=20,
            retrieval_mode="bm25",
            cache_dir="./cache",
            llm_model_name="Qwen/Qwen2.5-7B-Instruct",
            device=device
        )
        
        rag = LinearRAG(config)
        rag.load_corpus("LinearRAG/pubmed_qa", corpus_name="mirage", split="corpus")
        
        logging.info(f"[GPU {rank}] Graph building completed!")
        return True
        
    except Exception as e:
        logging.error(f"[GPU {rank}] Error building graph: {str(e)}")
        return False

def evaluate_single_dataset(rank, dataset_name):
    """在指定GPU上评估单个完整数据集"""
    try:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        
        logging.info(f"[GPU {rank}] Starting evaluation for {dataset_name} on {device}")
        
        # 初始化RAG（会加载已构建的图）
        config = LinearRAGConfig(
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            top_p=20,
            retrieval_mode="bm25",
            cache_dir="./cache",
            llm_model_name="Qwen/Qwen2.5-7B-Instruct",
            device=device
        )
        
        rag = LinearRAG(config)
        rag.load_corpus("LinearRAG/pubmed_qa", corpus_name="mirage", split="corpus")
        
        # 运行评估
        results = rag.run_qa_evaluation(
            dataset_name=dataset_name,
            split="test",
            save_dir=f"./results/{dataset_name}",
            batch_size=8,
            max_parallel=16
        )
        
        logging.info(f"[GPU {rank}] Completed {dataset_name}")
        logging.info(f"[GPU {rank}] LLM Accuracy: {results.get('llm_accuracy', 0):.4f}")
        logging.info(f"[GPU {rank}] Contain Accuracy: {results.get('contain_accuracy', 0):.4f}")
        
        return {
            'dataset': dataset_name,
            'gpu': rank,
            'results': results
        }
        
    except Exception as e:
        logging.error(f"[GPU {rank}] Error evaluating {dataset_name}: {str(e)}")
        return {
            'dataset': dataset_name,
            'gpu': rank,
            'error': str(e)
        }

def main():
    """主函数：先构建图，再并行评估"""
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    print("=" * 50)
    print("Multi-GPU Evaluation Pipeline")
    print("=" * 50)
    
    # 步骤1: 在GPU 0上构建图
    print("\n[Step 1] Building graph on GPU 0...")
    success = build_graph_once(rank=0)
    
    if not success:
        print("Failed to build graph. Exiting...")
        return
    
    print("[Step 1] Graph building completed!\n")
    
    # 步骤2: 在5个GPU上并行评估5个数据集
    print("[Step 2] Starting parallel evaluation on 5 GPUs...")
    print(f"Datasets to evaluate: {list(DATASETS.values())}\n")
    
    # 创建进程池
    processes = []
    results_queue = mp.Queue()
    
    for gpu_id, dataset_name in DATASETS.items():
        p = mp.Process(
            target=lambda q, r, d: q.put(evaluate_single_dataset(r, d)),
            args=(results_queue, gpu_id, dataset_name)
        )
        p.start()
        processes.append(p)
        print(f"Started evaluation of {dataset_name} on GPU {gpu_id}")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集结果
    all_results = []
    while not results_queue.empty():
        all_results.append(results_queue.get())
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("Final Results Summary")
    print("=" * 50)
    
    for result in sorted(all_results, key=lambda x: x['gpu']):
        if 'error' in result:
            print(f"\nGPU {result['gpu']} - {result['dataset']}: ERROR")
            print(f"  Error: {result['error']}")
        else:
            print(f"\nGPU {result['gpu']} - {result['dataset']}:")
            print(f"  LLM Accuracy: {result['results'].get('llm_accuracy', 0):.4f}")
            print(f"  Contain Accuracy: {result['results'].get('contain_accuracy', 0):.4f}")
    
    # 保存汇总结果
    summary_file = f"./results/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("./results", exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[Done] Summary saved to {summary_file}")
    print("=" * 50)

if __name__ == "__main__":
    main()
