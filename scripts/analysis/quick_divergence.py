#!/usr/bin/env python3
import numpy as np, networkx as nx, scipy.sparse as sp, os, json, time

def analyze(n, p, cb):
    name = f"G{n}_{cb}"
    G = nx.erdos_renyi_graph(n, p)
    nodes = list(G.nodes())
    for _ in range(int(n * cb)):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v): G.add_edge(u, v)
    adj = nx.adjacency_matrix(G)
    
    row_sums = np.array(adj.sum(axis=1)).flatten(); row_sums[row_sums==0] = 1
    P = sp.diags(1.0/row_sums) @ adj
    ppr = np.zeros((n,n,4)); Pk = sp.eye(n)
    for k in range(4): Pk = Pk @ P; ppr[:,:,k] = Pk.toarray()
    
    A = adj.toarray(); spse = np.zeros((n,n,4)); Ak = np.eye(n)
    for k in range(4): Ak = Ak @ A; spse[:,:,k] = Ak
    
    div = np.linalg.norm(ppr.flatten()/(np.linalg.norm(ppr.flatten())+1e-10) - spse.flatten()/(np.linalg.norm(spse.flatten())+1e-10))
    
    cycles = list(nx.simple_cycles(G.to_directed())); total = len(cycles)
    cd = total / (n*(n-1)*(n-2)/6 + 1e-10)
    return {'name':name, 'n':n, 'edges':adj.nnz, 'div':float(div), 'cd':float(cd), 'cycles':total}

print("PPR vs SPSE Divergence Analysis"); print("="*50)
results = [analyze(n, p, cb) for n, p, cb in [(20,0.08,0.1),(20,0.08,0.5),(20,0.2,0.1),(20,0.2,0.5),(25,0.12,0.3)]]
for r in results: print(f"{r['name']:8} | n={r['n']:2} | e={r['edges']:3} | D={r['div']:.4f} | CD={r['cd']:.6f} | cycles={r['cycles']}")
corr = np.corrcoef([r['cd'] for r in results], [r['div'] for r in results])[0,1]
print(f"\nCorrelation: {corr:.4f}")
os.makedirs('./figs/ppr_spse_divergence', exist_ok=True)
with open('./figs/ppr_spse_divergence/results.json','w') as f: json.dump({'results':results,'correlation':float(corr)},f,indent=2)
print("Saved to ./figs/ppr_spse_divergence/results.json")
