import numpy as np
import torch
import tqdm


def progress(data_iter):
    return tqdm.tqdm(data_iter, bar_format="{l_bar}{r_bar}")


def get_neighbors(train_data):
    neighbors = {}
    for head, tail in train_data:
        if head not in neighbors:
            neighbors[head] = set()
        neighbors[head].add(tail)
        if tail not in neighbors:
            neighbors[tail] = set()
        neighbors[tail].add(head)
    return neighbors


def get_all_nodes(data):
    all_nodes = set()
    for head, tail in data:
        all_nodes.add(head)
        all_nodes.add(tail)
    return all_nodes


def get_node_candidates(all_nodes, neighbors, node, skip_existing_candidates=True):
    if not skip_existing_candidates:
        return list(all_nodes)
    return list(all_nodes.difference(neighbors[node]))


def difference(ta, tb):
    vs, cs = torch.cat([ta, tb]).unique(return_counts=True)
    return vs[cs == 1]


default_value = None


def get_node_candidates_gpu(all_nodes, neighbors, node, skip_existing_candidates=True):
    if not skip_existing_candidates:
        return all_nodes
    return difference(all_nodes, neighbors.get(node, default_value))


def rank_data(data):
    _, idx, count = torch.unique(data, return_inverse=True, return_counts=True)
    return (torch.cumsum(count, 0) - 0.5 * count + 0.5)[idx]


def rank(node, true_candidate, node_reps, neighbors, all_nodes, device,
         edge_type, use_edge_type=True, skip_existing_candidates=True):
    global default_value
    if default_value is None:
        default_value = torch.tensor([]).to(device)
    if use_edge_type:
        node_tensor = node_reps(torch.LongTensor([node]).to(device),
                                torch.LongTensor([edge_type]).to(device)).view(-1, 1)
    else:
        node_tensor = node_reps(torch.LongTensor([node]).to(device)).view(-1, 1)
    true_candidate = torch.LongTensor([true_candidate]).to(device)
    candidates = get_node_candidates_gpu(all_nodes, neighbors, node, skip_existing_candidates=skip_existing_candidates)
    candidates = torch.cat([candidates, true_candidate])
    length = candidates.shape[0]
    if use_edge_type:
        candidate_tensors = node_reps(candidates, torch.LongTensor([edge_type]).to(device).repeat(length))
    else:
        candidate_tensors = node_reps(candidates)
    scores = torch.mm(candidate_tensors, node_tensor)
    negative_scores_numpy = -scores.view(1, -1)
    rank_ = rank_data(negative_scores_numpy)[0][-1].item()
    return rank_, length


def get_ranks(test_data, node_reps, neighbors, all_nodes, device, verbose=True, skip_existing_candidates=True):
    head_ranks = []
    tail_ranks = []
    head_lengths = []
    tail_lengths = []
    data_iter = tqdm.tqdm(test_data, bar_format='{l_bar}{r_bar}') if verbose else test_data
    for edge in data_iter:
        head_node, tail_node, edge_type = map(int, edge[:3])
        if head_node in all_nodes and tail_node in all_nodes:
            head_rank, head_length = rank(head_node, tail_node, node_reps, neighbors, all_nodes, device, edge_type,
                                          skip_existing_candidates=skip_existing_candidates)
            head_ranks.append(head_rank)
            head_lengths.append(head_length)

            tail_rank, tail_length = rank(tail_node, head_node, node_reps, neighbors, all_nodes, device, edge_type,
                                          skip_existing_candidates=skip_existing_candidates)
            tail_ranks.append(tail_rank)
            tail_lengths.append(tail_length)

    return head_ranks, tail_ranks, head_lengths, tail_lengths


def _f(_i, _edge, n_test, all_nodes, node_reps, neighbors, device, skip_existing_candidates):
    if _i % 5000 == 0:
        print(f'{_i}/{n_test}')
    _head_node, _tail_node, _edge_type = map(int, _edge[:3])
    assert _head_node in all_nodes and _tail_node in all_nodes
    _head_rank, _head_length = rank(_head_node, _tail_node, node_reps, neighbors, all_nodes, device, _edge_type,
                                    skip_existing_candidates=skip_existing_candidates)
    _tail_rank, _tail_length = rank(_tail_node, _head_node, node_reps, neighbors, all_nodes, device, _edge_type,
                                    skip_existing_candidates=skip_existing_candidates)
    return _head_rank, _head_length, _tail_rank, _tail_length


def get_rank_info(head_ranks, tail_ranks, head_lengths, tail_lengths, verbose=True):
    head_ranks_numpy = np.asarray(head_ranks)
    tail_ranks_numpy = np.asarray(tail_ranks)
    head_lengths_numpy = np.asarray(head_lengths)
    tail_lengths_numpy = np.asarray(tail_lengths)
    mean_head_rank = np.mean(head_ranks_numpy)
    mean_tail_rank = np.mean(tail_ranks_numpy)
    verbose and print(f'num_test: {head_lengths_numpy.shape[0]}; '
                      f'lengths mean (head/tail): {np.mean(head_lengths_numpy)} / {np.mean(tail_lengths_numpy)}')
    verbose and print(f'rank (head/tail): '
                      f'{mean_head_rank}±{np.var(head_ranks_numpy)} / {mean_tail_rank}±{np.var(tail_ranks_numpy)}')
    verbose and print(
        f'reverse rank mean (head/tail): {np.mean(1 / head_ranks_numpy)} / {np.mean(1 / tail_ranks_numpy)}')
    verbose and print(f'head_rank HITS 20/50/100: '
                      f'{(head_ranks_numpy <= 20).sum()}/'
                      f'{(head_ranks_numpy <= 50).sum()}/'
                      f'{(head_ranks_numpy <= 100).sum()}')
    verbose and print(f'tail_rank HITS 20/50/100: '
                      f'{(tail_ranks_numpy <= 20).sum()}/'
                      f'{(tail_ranks_numpy <= 50).sum()}/'
                      f'{(tail_ranks_numpy <= 100).sum()}')
    return mean_head_rank, mean_tail_rank, head_ranks_numpy, tail_ranks_numpy, head_lengths_numpy, tail_lengths_numpy


def get_metrics(edges, model, neighbors, all_nodes, device):
    with torch.no_grad():
        head_ranks, tail_ranks, head_lengths, tail_lengths = get_ranks(
            edges, model.node_embeddings, neighbors, all_nodes, device)
        _, _, head_ranks_numpy, tail_ranks_numpy, head_lengths_numpy, tail_lengths_numpy = get_rank_info(
            head_ranks, tail_ranks, head_lengths, tail_lengths)
        print('==========================================')
        print('MRR:', (np.mean(1 / head_ranks_numpy) + np.mean(1 / tail_ranks_numpy)) / 2)
        print('HitRate@100:',
              ((head_ranks_numpy <= 100).sum() + (tail_ranks_numpy <= 100).sum()) / head_lengths_numpy.shape[0] / 2)
        print('HitRate@50:',
              ((head_ranks_numpy <= 50).sum() + (tail_ranks_numpy <= 50).sum()) / head_lengths_numpy.shape[0] / 2)
        print('HitRate@20:',
              ((head_ranks_numpy <= 20).sum() + (tail_ranks_numpy <= 20).sum()) / head_lengths_numpy.shape[0] / 2)
