from typing import List, Tuple

import torch
import torch.nn.functional as F
import networkx as nx


def extract_motion_hierarchy(motion_structure_logits: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Extract the motion hierarchy from the motion structure logits using NetworkX's 
    maximum_spanning_arborescence (which implements Edmonds' algorithm).
    
    Args:
        motion_structure_logits: (N, N) tensor where motion_structure_logits[i,j] 
                               represents the logit for directed edge from i to j
    
    Returns:
        List of (parent, child) tuples representing the directed spanning tree
    """
    weights = F.logsigmoid(motion_structure_logits).detach().cpu().numpy()
    
    n = weights.shape[0]
    
    if n <= 1:
        return []
    
    G = nx.DiGraph()

    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=weights[i, j])

    arborescence = nx.maximum_spanning_arborescence(G, attr='weight')

    result = []
    for p, c in arborescence.edges():
        result.append((p, c))

    return result
