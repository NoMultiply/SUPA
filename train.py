import argparse
import os
import pickle
import random

from SUPADataset import SUPADataset
from SUPAModel import SUPAModel
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for SUPA')
    parser.add_argument('--dataset', type=str, help='the dataset', default='TianChi')

    parser.add_argument('--walk_length', type=int, help='walk length', default=3)
    parser.add_argument('--n_walks', type=int, help='the num of walks', default=5)
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--embedding_size', type=int, help='the size of node embedding', default=128)
    parser.add_argument('--n_negative', type=int, help='the negative sample num', default=5)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.003)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.0001)
    parser.add_argument('--h_delta', type=float, help='the tau threshold', default=0.3)
    parser.add_argument('--gpu_id', type=int, help='which gpu to use', default=2)
    parser.add_argument('--patient', type=int, help='early stop patient', default=3)
    parser.add_argument('--max_iter', type=int, help='max iterations per batch', default=None)
    parser.add_argument('--max_neighbors', type=int, help='max neighbors to keep', default=10)
    parser.add_argument('--valid_interval', type=int, help='valid per iterations', default=None)
    parser.add_argument('--train_on_valid', action='store_true', help='train on the valid set')
    parser.add_argument('--n_valid', type=int, help='number of valid edges', default=100)
    parser.add_argument('--time_agg', type=str, help='time aggregations gap', default='604800')
    parser.add_argument('--regen_walks', action='store_true', help='regenerate walks')
    parser.add_argument('--schema', type=str, help='the metapath instances, eg, 0-1-0,1-0-1', default=None)
    args = parser.parse_args()
    args.time_agg = eval(args.time_agg)
    args.data = f'data/{args.dataset}'
    args.model = f'model/{args.dataset}'
    return args


def get_batches(data_list, batch_size):
    batches = []
    for i in range(0, len(data_list[0]), batch_size):
        batch_data = tuple(data[i:i + batch_size] for data in data_list)
        batches.append(batch_data)
    return batches


def generate_walks(train_edges, n_walks, walk_length):
    def _go_walk(_s):
        _walks = []
        for _ in range(n_walks):
            _walk = [_s]
            for _ in range(walk_length):
                _walk.append(random.choice(list(neighbors[_walk[-1]])))
            _walks.append(_walk[1:])
        return _walks

    walks = []
    neighbors = {}
    nodes = set()
    print('Generating walks')
    for edge in tqdm.tqdm(train_edges, bar_format='{l_bar}{r_bar}'):
        u, v = map(int, edge[:2])
        if u not in neighbors:
            neighbors[u] = set()
        if v not in neighbors:
            neighbors[v] = set()
        neighbors[u].add(v)
        neighbors[v].add(u)
        nodes.add(u)
        nodes.add(v)
        walks.append([_go_walk(u), _go_walk(v)])
    return walks


def generate_walks_with_interaction_times_edge_types(train_edges, n_walks, walk_length, node_types, schema_str):
    node_schemas = {}
    if schema_str is not None:
        schemas = [list(map(int, item.split('-'))) for item in schema_str.split(',')]
        for schema in schemas:
            assert schema[0] == schema[-1], 'Schema must be symmetric'
            if schema[0] not in node_schemas:
                node_schemas[schema[0]] = []
            node_schemas[schema[0]].append(schema)

    def _go_walk(_s):
        _walks = []
        _interaction_times = []
        _edge_types = []
        for _ in range(n_walks):
            _walk = [_s]
            _interaction_time = []
            _edge_type = []
            if schema_str is None:
                for _ in range(walk_length):
                    _current = _walk[-1]
                    _next = random.choice(list(neighbors[_current].keys()))
                    _walk.append(_next)
                    _interaction_time.append(neighbors[_current][_next][0])
                    _edge_type.append(neighbors[_current][_next][1])
            else:
                assert node_types[_s] in node_schemas, \
                    f'No schema found for node type "{node_types[_s]}"'
                _schema = random.choice(node_schemas[node_types[_s]])
                for _ in range(walk_length):
                    _current = _walk[-1]
                    _next = random.choice([x for x in neighbors[_current].keys() if
                                           node_types[x] == _schema[len(_walk) % (len(_schema) - 1)]])
                    _walk.append(_next)
                    _interaction_time.append(neighbors[_current][_next][0])
                    _edge_type.append(neighbors[_current][_next][1])
            _walks.append(_walk[1:])
            _interaction_times.append(_interaction_time)
            _edge_types.append(_edge_type)
        return _walks, _interaction_times, _edge_types

    walks = []
    interaction_times = []
    edge_types = []
    neighbors = {}
    nodes = set()
    print('Generating walks with interaction times')
    for edge in tqdm.tqdm(train_edges, bar_format='{l_bar}{r_bar}'):
        u, v, e, t = map(int, edge)
        if u not in neighbors:
            neighbors[u] = {}
        if v not in neighbors:
            neighbors[v] = {}
        neighbors[u][v] = (t, e)
        neighbors[v][u] = (t, e)
        nodes.add(u)
        nodes.add(v)
        u_walks, u_interaction_times, u_edge_types = _go_walk(u)
        v_walks, v_interaction_times, v_edge_types = _go_walk(v)
        walks.append([u_walks, v_walks])
        interaction_times.append([u_interaction_times, v_interaction_times])
        edge_types.append([u_edge_types, v_edge_types])
    return walks, interaction_times, edge_types


def generate_walks_with_interaction_times_edge_types_with_drop(train_edges, n_walks, walk_length, node_types,
                                                               schema_str, max_neighbors):
    node_schemas = {}
    if schema_str is not None:
        schemas = [list(map(int, item.split('-'))) for item in schema_str.split(',')]
        for schema in schemas:
            assert schema[0] == schema[-1], 'Schema must be symmetric'
            if schema[0] not in node_schemas:
                node_schemas[schema[0]] = []
            node_schemas[schema[0]].append(schema)

    def _go_walk(_s):
        _walks = []
        _interaction_times = []
        _edge_types = []
        for _ in range(n_walks):
            _walk = [_s]
            _interaction_time = []
            _edge_type = []
            if schema_str is None:
                for _ in range(walk_length):
                    _current = _walk[-1]
                    _choice = random.choice(neighbors[_current])
                    _walk.append(_choice[0])
                    _interaction_time.append(_choice[1])
                    _edge_type.append(_choice[2])
            else:
                assert node_types[_s] in node_schemas, \
                    f'No schema found for node type "{node_types[_s]}"'
                _schema = random.choice(node_schemas[node_types[_s]])
                for _ in range(walk_length):
                    _current = _walk[-1]
                    _choice = random.choice([x for x in neighbors[_current] if
                                             node_types[x[0]] == _schema[len(_walk) % (len(_schema) - 1)]])
                    _walk.append(_choice[0])
                    _interaction_time.append(_choice[1])
                    _edge_type.append(_choice[2])
            _walks.append(_walk[1:])
            _interaction_times.append(_interaction_time)
            _edge_types.append(_edge_type)
        return _walks, _interaction_times, _edge_types

    walks = []
    interaction_times = []
    edge_types = []
    neighbors = {}
    nodes = set()
    print('Generating walks with interaction times')
    for edge in tqdm.tqdm(train_edges, bar_format='{l_bar}{r_bar}'):
        u, v, e, t = map(int, edge)
        if u not in neighbors:
            neighbors[u] = []
        if v not in neighbors:
            neighbors[v] = []
        neighbors[u].append((v, t, e))
        if len(neighbors[u]) > max_neighbors:
            neighbors[u].pop(0)
        neighbors[v].append((u, t, e))
        if len(neighbors[v]) > max_neighbors:
            neighbors[v].pop(0)
        nodes.add(u)
        nodes.add(v)
        u_walks, u_interaction_times, u_edge_types = _go_walk(u)
        v_walks, v_interaction_times, v_edge_types = _go_walk(v)
        walks.append([u_walks, v_walks])
        interaction_times.append([u_interaction_times, v_interaction_times])
        edge_types.append([u_edge_types, v_edge_types])
    return walks, interaction_times, edge_types


def valid_model(valid_edges, model, neighbors, all_nodes, device):
    with torch.no_grad():
        head_ranks, tail_ranks, head_lengths, tail_lengths = \
            get_ranks(valid_edges, model.node_embeddings,
                      neighbors, all_nodes, device, verbose=False, skip_existing_candidates=False)
    _, _, head_ranks_numpy, tail_ranks_numpy, _, _ = get_rank_info(
        head_ranks, tail_ranks, head_lengths, tail_lengths, verbose=False)
    return (np.mean(1 / head_ranks_numpy) + np.mean(1 / tail_ranks_numpy)) / 2


def train(args):
    dataset = SUPADataset(args.data)
    dataset.print_info()

    model_save_dir = args.model + f'_es_{args.embedding_size}_bs_{args.batch_size}_lr_{args.learning_rate}' \
                                  f'_wd_{args.weight_decay}_wl_{args.walk_length}_nl_{args.n_walks}' \
                                  f'_nn_{args.n_negative}'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = SUPAModel(device, dataset.n_nodes, args.embedding_size, dataset.n_node_types, dataset.n_edge_types,
                      dataset.node_types, dataset.edge_types, args.n_negative, args.time_agg)
    model.train()

    optimizer_params = {
        'params': model.parameters(),
        'lr': args.learning_rate
    }
    if args.weight_decay:
        optimizer_params['weight_decay'] = args.weight_decay
    optimizer = torch.optim.Adam(**optimizer_params)

    all_nodes = torch.tensor(list(get_all_nodes(dataset.train_edges[:, [0, 1]]))).to(device)
    neighbors = get_neighbors(dataset.train_edges[:, [0, 1]])
    for k in neighbors:
        neighbors[k] = torch.tensor(list(neighbors[k])).to(device)

    walks_root = 'walks'
    if not os.path.exists(walks_root):
        os.makedirs(walks_root)

    walks_filename = os.path.join(
        walks_root, args.data.replace('/', '_') + f'_walks_with_interaction_times_edge_types'
                                                  f'_{args.walk_length}_{args.n_walks}_{args.schema}_'
                                                  f'{args.train_on_valid}.pkl')
    if args.train_on_valid:
        dataset.train_edges = np.concatenate([dataset.train_edges, dataset.valid_edges], axis=0)
    if not os.path.exists(walks_filename) or args.regen_walks:
        walks, interaction_times, edge_types = generate_walks_with_interaction_times_edge_types(
            dataset.train_edges, args.n_walks, args.walk_length, dataset.node_types, args.schema)
        print('Saving walks to', walks_filename)
        with open(walks_filename, 'wb') as out:
            pickle.dump((walks, interaction_times, edge_types), out)
    print('Load walks from', walks_filename)
    with open(walks_filename, 'rb') as fin:
        walks, interaction_times, edge_types = pickle.load(fin)
    batches = get_batches((dataset.train_edges, walks, interaction_times, edge_types), args.batch_size)

    data_iter = progress(batches)
    for i, batch_data in enumerate(data_iter):
        valid_edges = batches[i][0][-args.n_valid:]
        model.train_batch(batch_data, device, args.patient, args.max_iter, optimizer, args.valid_interval,
                          valid_model, neighbors, data_iter, i, valid_edges, args.h_delta)
    get_metrics(dataset.valid_edges, model, neighbors, all_nodes, device)
    model_save_filename = model_save_dir + '/' + 'model_final.pt'
    torch.save(model.state_dict(), model_save_filename)

    model.load_state_dict(torch.load(model_save_filename))
    model.eval()
    get_metrics(dataset.test_edges, model, neighbors, all_nodes, device)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
