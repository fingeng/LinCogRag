#!/usr/bin/env python3
"""
下载PubMedQA测试集对应的PubMed论文摘要
然后添加到现有的corpus中
"""

import json
import os
import time
from Bio import Entrez
from tqdm import tqdm
import argparse

# 设置Entrez邮箱（PubMed API要求）
Entrez.email = "your_email@example.com"  # 请修改为你的邮箱

def load_pubmedqa_pmids(data_file='MIRAGE/rawdata/pubmedqa/data/test_set.json'):
    """加载PubMedQA测试集的所有PMID"""
    print(f"Loading PubMedQA PMIDs from {data_file}...")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    pmids = list(data.keys())
    print(f"Found {len(pmids)} PMIDs")
    
    return pmids, data

def fetch_pubmed_abstract(pmid, retry=3):
    """
    从PubMed下载单个论文的摘要
    
    返回格式:
    {
        'pmid': str,
        'title': str,
        'abstract': str,
        'structured_abstract': dict or None,  # 如果是结构化摘要
        'journal': str,
        'year': str,
    }
    """
    for attempt in range(retry):
        try:
            # 使用Entrez.efetch获取摘要
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="abstract",
                retmode="xml"
            )
            
            from Bio import Medline
            records = Entrez.read(handle)
            handle.close()
            
            if not records or 'PubmedArticle' not in records:
                print(f"  ⚠️  PMID {pmid}: No article found")
                return None
            
            article = records['PubmedArticle'][0]
            medline_citation = article['MedlineCitation']
            article_data = medline_citation['Article']
            
            # 提取基本信息
            pmid_str = str(medline_citation['PMID'])
            title = article_data.get('ArticleTitle', '')
            
            # 提取摘要
            abstract_text = ''
            structured_abstract = None
            
            if 'Abstract' in article_data:
                abstract = article_data['Abstract']
                
                if 'AbstractText' in abstract:
                    abstract_texts = abstract['AbstractText']
                    
                    # 检查是否是结构化摘要
                    if isinstance(abstract_texts, list):
                        structured_abstract = {}
                        full_abstract = []
                        
                        for text_item in abstract_texts:
                            if hasattr(text_item, 'attributes') and 'Label' in text_item.attributes:
                                label = text_item.attributes['Label']
                                content = str(text_item)
                                structured_abstract[label] = content
                                full_abstract.append(f"{label}: {content}")
                            else:
                                content = str(text_item)
                                full_abstract.append(content)
                        
                        abstract_text = ' '.join(full_abstract)
                    else:
                        abstract_text = str(abstract_texts)
            
            # 提取期刊和年份
            journal = ''
            year = ''
            
            if 'Journal' in article_data:
                journal_info = article_data['Journal']
                if 'Title' in journal_info:
                    journal = journal_info['Title']
                
                if 'JournalIssue' in journal_info:
                    issue = journal_info['JournalIssue']
                    if 'PubDate' in issue:
                        pub_date = issue['PubDate']
                        year = pub_date.get('Year', '')
            
            return {
                'pmid': pmid_str,
                'title': title,
                'abstract': abstract_text,
                'structured_abstract': structured_abstract,
                'journal': journal,
                'year': year,
            }
            
        except Exception as e:
            if attempt < retry - 1:
                print(f"  ⚠️  PMID {pmid} attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(2)
            else:
                print(f"  ❌ PMID {pmid} failed after {retry} attempts: {e}")
                return None
    
    return None

def download_all_abstracts(pmids, output_file='pubmedqa_abstracts.json', batch_size=10):
    """批量下载所有论文摘要"""
    
    # 检查是否已有部分下载
    downloaded = {}
    if os.path.exists(output_file):
        print(f"Found existing file {output_file}, loading...")
        with open(output_file, 'r') as f:
            downloaded = json.load(f)
        print(f"Already downloaded {len(downloaded)} abstracts")
    
    # 过滤已下载的
    remaining_pmids = [p for p in pmids if p not in downloaded]
    print(f"Need to download {len(remaining_pmids)} abstracts")
    
    if not remaining_pmids:
        print("All abstracts already downloaded!")
        return downloaded
    
    # 批量下载
    print(f"\nDownloading abstracts (batch size: {batch_size})...")
    
    for i in tqdm(range(0, len(remaining_pmids), batch_size)):
        batch = remaining_pmids[i:i+batch_size]
        
        for pmid in batch:
            result = fetch_pubmed_abstract(pmid)
            
            if result:
                downloaded[pmid] = result
                # print(f"  ✅ PMID {pmid}: {result['title'][:60]}...")
            
            # PubMed API限制: 每秒最多3个请求
            time.sleep(0.4)
        
        # 每个batch保存一次
        with open(output_file, 'w') as f:
            json.dump(downloaded, f, indent=2)
        
        # 每10个batch休息一下
        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"\n  Saved checkpoint: {len(downloaded)} abstracts")
            time.sleep(2)
    
    print(f"\n✅ Download complete! Total: {len(downloaded)} abstracts")
    print(f"   Saved to: {output_file}")
    
    return downloaded

def create_pubmedqa_chunks(abstracts_file='pubmedqa_abstracts.json', 
                           original_data_file='MIRAGE/rawdata/pubmedqa/data/test_set.json'):
    """
    将下载的摘要转换为chunks格式
    每个PMID创建1个chunk（完整摘要）
    """
    
    print(f"\nCreating chunks from abstracts...")
    
    with open(abstracts_file, 'r') as f:
        abstracts = json.load(f)
    
    with open(original_data_file, 'r') as f:
        pubmedqa_data = json.load(f)
    
    chunks = []
    
    for pmid, abstract_info in abstracts.items():
        if not abstract_info or not abstract_info.get('abstract'):
            print(f"  ⚠️  Skipping PMID {pmid}: No abstract")
            continue
        
        # 构建完整文本: Title + Abstract
        text = f"{abstract_info['title']} {abstract_info['abstract']}"
        
        # 添加chunk
        chunk = {
            'pmid': pmid,
            'text': text,
            'title': abstract_info['title'],
            'abstract': abstract_info['abstract'],
            'structured_abstract': abstract_info.get('structured_abstract'),
            'journal': abstract_info.get('journal', ''),
            'year': abstract_info.get('year', ''),
            'source': 'pubmedqa',
        }
        
        # 如果有原始question，也加上
        if pmid in pubmedqa_data:
            chunk['question'] = pubmedqa_data[pmid]['QUESTION']
            chunk['answer'] = pubmedqa_data[pmid]['final_decision']
        
        chunks.append(chunk)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    return chunks

def append_to_existing_corpus(chunks, 
                               corpus_dir='dataset/pubmed/chunk',
                               output_file='pubmed_with_pubmedqa.jsonl'):
    """
    将PubMedQA chunks追加到现有corpus
    """
    
    print(f"\nAppending PubMedQA chunks to existing corpus...")
    
    # 读取现有corpus
    original_file = os.path.join(corpus_dir, 'pubmed.jsonl')
    
    if not os.path.exists(original_file):
        print(f"  ⚠️  Original corpus not found: {original_file}")
        print(f"  Creating new corpus file...")
        existing_chunks = []
    else:
        existing_chunks = []
        with open(original_file, 'r') as f:
            for line in f:
                existing_chunks.append(json.loads(line))
        print(f"  Original corpus: {len(existing_chunks)} chunks")
    
    # 合并
    all_chunks = existing_chunks + chunks
    print(f"  Total chunks: {len(all_chunks)}")
    
    # 保存
    output_path = os.path.join(corpus_dir, output_file)
    with open(output_path, 'w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"✅ Saved augmented corpus to: {output_path}")
    print(f"   Total: {len(all_chunks)} chunks ({len(existing_chunks)} original + {len(chunks)} PubMedQA)")
    
    return output_path

def analyze_downloaded_abstracts(abstracts_file='pubmedqa_abstracts.json'):
    """分析下载的摘要质量"""
    
    print(f"\n" + "="*80)
    print("Downloaded Abstracts Analysis")
    print("="*80)
    
    with open(abstracts_file, 'r') as f:
        abstracts = json.load(f)
    
    total = len(abstracts)
    has_abstract = sum(1 for a in abstracts.values() if a and a.get('abstract'))
    has_structured = sum(1 for a in abstracts.values() if a and a.get('structured_abstract'))
    
    print(f"\nTotal PMIDs: {total}")
    print(f"Has abstract: {has_abstract} ({has_abstract/total*100:.1f}%)")
    print(f"Structured abstract: {has_structured} ({has_structured/total*100:.1f}%)")
    
    # 统计摘要长度
    lengths = [len(a['abstract']) for a in abstracts.values() if a and a.get('abstract')]
    if lengths:
        print(f"\nAbstract length (characters):")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {sum(lengths)/len(lengths):.0f}")
    
    # 显示几个样例
    print(f"\n" + "="*80)
    print("Sample Abstracts")
    print("="*80)
    
    for i, (pmid, info) in enumerate(list(abstracts.items())[:3]):
        if not info:
            continue
        print(f"\nPMID: {pmid}")
        print(f"Title: {info.get('title', 'N/A')[:100]}...")
        print(f"Journal: {info.get('journal', 'N/A')}")
        print(f"Year: {info.get('year', 'N/A')}")
        
        if info.get('structured_abstract'):
            print(f"Structured sections: {list(info['structured_abstract'].keys())}")
        
        if info.get('abstract'):
            print(f"Abstract (first 200 chars): {info['abstract'][:200]}...")

def main():
    parser = argparse.ArgumentParser(description='Download PubMedQA abstracts and augment corpus')
    parser.add_argument('--email', type=str, default='your_email@example.com',
                        help='Your email for PubMed API')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for downloading')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download abstracts, do not create chunks')
    parser.add_argument('--output', type=str, default='pubmedqa_abstracts.json',
                        help='Output file for abstracts')
    
    args = parser.parse_args()
    
    # 设置邮箱
    Entrez.email = args.email
    
    # Step 1: 加载PMIDs
    pmids, pubmedqa_data = load_pubmedqa_pmids()
    
    # Step 2: 下载摘要
    abstracts = download_all_abstracts(
        pmids, 
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    # Step 3: 分析下载结果
    analyze_downloaded_abstracts(args.output)
    
    if args.download_only:
        print("\n✅ Download complete (--download-only mode)")
        return
    
    # Step 4: 创建chunks
    chunks = create_pubmedqa_chunks(args.output)
    
    # Step 5: 保存chunks
    chunks_file = 'pubmedqa_chunks.json'
    with open(chunks_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"✅ Saved chunks to: {chunks_file}")
    
    # Step 6: 追加到现有corpus（可选）
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print(f"""
1. ✅ Downloaded {len(abstracts)} abstracts
2. ✅ Created {len(chunks)} chunks
3. ✅ Saved to {chunks_file}

To use these abstracts:

Option A: Append to existing corpus (recommended)
  python3 << 'EOF'
import json
import os

# 读取原始corpus
with open('dataset/pubmed/chunk/pubmed.jsonl', 'r') as f:
    original = [json.loads(line) for line in f]

# 读取PubMedQA chunks
with open('{chunks_file}', 'r') as f:
    pubmedqa_chunks = json.load(f)

# 转换为相同格式
for chunk in pubmedqa_chunks:
    original.append({{'text': chunk['text'], 'source': 'pubmedqa'}})

# 保存
with open('dataset/pubmed/chunk/pubmed_augmented.jsonl', 'w') as f:
    for chunk in original:
        f.write(json.dumps(chunk) + '\\n')

print(f"Total chunks: {{len(original)}}")
EOF

Option B: Build separate PubMedQA corpus
  - Modify run.py to add pubmedqa dataset support
  - Build graph from pubmedqa_chunks.json
  - Test retrieval on this specialized corpus

Then rebuild the graph:
  python run.py --dataset_name pubmed \\
                --dataset pubmedqa \\
                --mirage_dataset pubmedqa \\
                --build_graph
""")

if __name__ == '__main__':
    main()
