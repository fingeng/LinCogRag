# filepath: /home/maoxy23/projects/LinearRAG/scripts/analyze_graph.py
"""
Analyze LinearRAG graph structure
"""
import igraph as ig
from collections import Counter

def analyze_graph(graphml_path):
    """Load and analyze graph"""
    print("="*70)
    print("LinearRAG Graph Analysis")
    print("="*70)
    
    # Load graph
    print("\n📊 Loading graph...")
    g = ig.Graph.Read_GraphML(graphml_path)
    
    # Basic statistics
    print(f"\n1. Basic Statistics:")
    print(f"   Total nodes: {g.vcount():,}")
    print(f"   Total edges: {g.ecount():,}")
    print(f"   Graph density: {g.density():.6f}")
    
    # Node type distribution
    print(f"\n2. Node Type Distribution:")
    node_types = Counter()
    for v in g.vs:
        name = v['name']
        if name.startswith('entity-'):
            node_types['Entity'] += 1
        elif name.startswith('passage-'):
            node_types['Passage'] += 1
        elif name.startswith('sentence-'):
            node_types['Sentence'] += 1
        else:
            node_types['Unknown'] += 1
    
    for ntype, count in node_types.items():
        pct = count / g.vcount() * 100
        print(f"   {ntype:15s}: {count:6,} ({pct:5.2f}%)")
    
    # Edge weight statistics
    print(f"\n3. Edge Weight Statistics:")
    if g.ecount() > 0:
        weights = g.es['weight']
        print(f"   Mean weight: {sum(weights)/len(weights):.4f}")
        print(f"   Min weight:  {min(weights):.4f}")
        print(f"   Max weight:  {max(weights):.4f}")
        print(f"   Median weight: {sorted(weights)[len(weights)//2]:.4f}")
    
    # Degree distribution
    print(f"\n4. Node Degree Distribution:")
    degrees = g.degree()
    print(f"   Mean degree: {sum(degrees)/len(degrees):.2f}")
    print(f"   Max degree:  {max(degrees)}")
    print(f"   Min degree:  {min(degrees)}")
    
    # Sample nodes
    print(f"\n5. Sample Nodes (first 5):")
    for i, v in enumerate(g.vs[:5]):
        print(f"\n   Node {i}:")
        print(f"     ID: {v['name'][:40]}...")
        print(f"     Content: {v['content'][:60]}...")
        print(f"     Degree: {v.degree()}")
    
    # Sample edges
    print(f"\n6. Sample Edges (first 5):")
    for i, e in enumerate(g.es[:5]):
        source = g.vs[e.source]
        target = g.vs[e.target]
        print(f"\n   Edge {i}:")
        print(f"     From: {source['content'][:30]}...")
        print(f"     To:   {target['content'][:30]}...")
        print(f"     Weight: {e['weight']:.4f}")
    
    # Connected components
    print(f"\n7. Graph Connectivity:")
    components = g.connected_components()
    print(f"   Number of connected components: {len(components)}")
    print(f"   Largest component size: {max(len(c) for c in components)}")
    
    return g

if __name__ == '__main__':
    graphml_path = 'import/pubmed_mirage_mmlu/LinearRAG.graphml'
    g = analyze_graph(graphml_path)
    print("\n" + "="*70)