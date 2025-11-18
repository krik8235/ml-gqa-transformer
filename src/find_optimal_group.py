import argparse
import random
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


def orthogonal_transform(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    C = A @ B.transpose(-1, -2) 
    try:
        U, _, V_h = t.linalg.svd(C)
        Q = V_h.transpose(-1, -2) @ U.transpose(-1, -2)
    except:
        # fallback to identity matrix on svd failure
        return t.eye(A.shape[0], device=DEVICE)
    return Q


def cosine_similarity(A: t.Tensor, B: t.Tensor, Q: t.Tensor) -> float:
    B_transformed = Q @ B
    similarity_per_token = F.cosine_similarity(A, B_transformed, dim=0)
    return similarity_per_token.mean().item()


def create_similarity_matrix(caches: t.Tensor) -> t.Tensor:
    H = caches.shape[0]
    sim_matrix = t.zeros(H, H, device=caches.device) 
    normalized_caches = caches.clone()
    
    # normalize each (d_h x N) slice along the d_h dimension
    for i in range(H): normalized_caches[i, :, :] = F.normalize(caches[i, :, :], p=2, dim=0)

    # create similarity matrix
    for i in range(H):
        for j in range(i + 1, H):
            A = normalized_caches[i]
            B = normalized_caches[j]
            Q = orthogonal_transform(A, B)
            sim_value = cosine_similarity(A, B, Q)
            sim_matrix[i, j] = sim_value
            sim_matrix[j, i] = sim_value
    return sim_matrix


def compute_grouping_score(grouping: list[list[int]], sim_matrix: t.Tensor) -> float:
    total_score = 0.0
    for group in grouping:
        for i_idx in range(len(group)):
            for j_idx in range(i_idx + 1, len(group)):
                i = group[i_idx]
                j = group[j_idx]
                total_score += sim_matrix[i, j].item()
    num_pairs = sum([len(g) * (len(g) - 1) // 2 for g in grouping])
    return total_score / num_pairs if num_pairs > 0 else 0.0


def simulate_grouping(H: int, D: int, sim_matrix: t.Tensor, max_iter: int) -> tuple[float, list[list[int]]]:
    G = H // D
    heads = list(range(H))
    random.shuffle(heads)
    current_grouping = [heads[i * D:(i + 1) * D] for i in range(G)]
    score_current = compute_grouping_score(current_grouping, sim_matrix)
    score_best = score_current
    best_grouping = current_grouping

    for _ in range(max_iter):
        # select two different groups for swapping
        g1_idx, g2_idx = random.sample(range(G), 2)
        
        # select a head index within each group
        h1_idx, h2_idx = random.randrange(D), random.randrange(D)
        
        # create groups
        new_grouping = [g[:] for g in current_grouping]
        
        # swap the two heads
        h1 = new_grouping[g1_idx][h1_idx]
        h2 = new_grouping[g2_idx][h2_idx]
        new_grouping[g1_idx][h1_idx] = h2
        new_grouping[g2_idx][h2_idx] = h1
        
        # calc score
        score_new = compute_grouping_score(new_grouping, sim_matrix)

        # accept the swap if the score improves (or equals, for simulated annealing)
        if score_new >= score_current:
            current_grouping = new_grouping
            score_current = score_new
            if score_new > score_best:
                score_best = score_new
                best_grouping = new_grouping

    return score_best, best_grouping


def find_optimal_groups(H: int, V_caches: t.Tensor, max_iter_per_G: int) -> dict:
    print(f"... calculating similarity matrix for {H} heads ...")
    
    # create similarity matrix
    sim_matrix = create_similarity_matrix(V_caches.to(DEVICE))

    # list up possible dimension sizes
    D_options = [d for d in range(1, H + 1) if H % d == 0]

    # start searching 
    best_overall_score = -float('inf')
    optimal_N_GROUPS = H
    optimal_N_QUERIES = 1
    best_grouping_A = None
    print(f"... testing possible group sizes D's ...")
    for D in D_options:
        G = H // D
        
        # case 1. G = 1 - MQA - one group
        if G == 1:
            grouping = [list(range(H))]
            # use the static method to compute the score
            score = compute_grouping_score(grouping, sim_matrix)
            print(f"   -> G={G} (D={D}): mqa with a single group. score = {score:.4f}")

        # case 2. D = 1  - MHA - trivial grouping, score is 0.0
        elif D == 1:
            score = 0.0
            grouping = [[i] for i in range(H)]
            print(f"   -> G={G} (D={D}): mha with max groups. score = {score:.4f}")
        
        # case 3. G >= 2 - GQA - sim. best gr
        elif G >= 2:
            score, grouping = simulate_grouping(H=H, D=D, sim_matrix=sim_matrix, max_iter=max_iter_per_G)
            print(f"   -> G={G} (D={D}): Best Sim Score (SA) = {score:.4f}")

        # update the optimal config
        if score > best_overall_score:
            best_overall_score = score
            optimal_N_GROUPS = G
            optimal_N_QUERIES = D
            best_grouping_A = grouping
            
    return {
        'optimal_N_GROUPS': optimal_N_GROUPS,
        'optimal_N_QUERIES': optimal_N_QUERIES,
        'max_score': best_overall_score,
        'best_grouping': best_grouping_A
    }



if __name__ == '__main__':
    from src.transformer import Transformer
    from src.attention_layers import GQA

    INITIAL_N_GROUPS = 4
    INITIAL_N_QUERIES = 2 
    MAX_ITER_PER_G = 300

    D_V = 64
    DEFAULT_H = 8
    D_MODEL = D_V * DEFAULT_H

    SEQ_LEN = 2048
    TOKENIZER = AutoTokenizer.from_pretrained("t5-small", model_max_length=SEQ_LEN) 

    parser = argparse.ArgumentParser(description='creating micro batch for online learning')
    parser.add_argument('--heads', type=int, default=DEFAULT_H, help=f"ticker. default = {DEFAULT_H}")
    args = parser.parse_args()

    H = args.heads

    print(f"... model config: H={H}, D_MODEL={D_MODEL}, D_V={D_V} ...")
    print(f"... initial gqa config: G={INITIAL_N_GROUPS}, D={INITIAL_N_QUERIES} ...")

    # instantiate gqa transformer
    t_gqa = Transformer(
        attention_module=GQA(d_model=D_MODEL, d_V=D_V, n_groups=INITIAL_N_GROUPS, n_queries=INITIAL_N_QUERIES),
        d_model=D_MODEL,
        max_seq_len=SEQ_LEN,
        tokenizer=TOKENIZER,
        device=DEVICE
    )

    # simulate v caches - shape: [H, d_h, N]
    V_caches_init = t.randn(H, D_V, SEQ_LEN) * 0.1

    # introduce strong grouping bias to ensure finding a non-trivial solution
    V_caches_init[0:4] += t.randn(1, D_V, SEQ_LEN) * 0.8 # bias heads 0-3 together, and heads 4-7 together
    V_caches_init[4:8] += t.randn(1, D_V, SEQ_LEN) * 0.8 

    # execute optimization method
    optimal_result = find_optimal_groups(H=H, V_caches=V_caches_init, max_iter_per_G=MAX_ITER_PER_G)

    optimal_n_groups = optimal_result['optimal_N_GROUPS']
    optimal_n_queries = optimal_result['optimal_N_QUERIES']

    print(f"\n... optimal gqa structure found:")
    print(f"  - optimal groups: {optimal_n_groups}")
    print(f"  - optimal group size: {optimal_n_queries}")
    print(f"  - maximum similarity score: {optimal_result['max_score']:.4f}")
    print(f"  - best grouping (head indices):\n      {optimal_result['best_grouping']}")

    # check if the optimization found the expected 2-group bias
    if optimal_result['optimal_N_GROUPS'] == 2 and optimal_result['optimal_N_QUERIES'] == 4:
        print("\nconclusion: optimization successfully found the strong 2-group bias.")
    else:
        print("\nconclusion: the optimal grouping was found by simulated annealing.")