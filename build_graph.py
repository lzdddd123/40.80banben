import argparse
import os
import pickle
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - local env fallback
    def tqdm(iterable, *args, **kwargs):
        return iterable

from dataset_utils import infer_num_nodes_from_sequences, resolve_dataset_dir


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='Tmall',
        help='Tmall/RetailRocket/lastfm/diginetica/yoochoose1_64/yoochoose1_4',
    )
    parser.add_argument('--sample_num', type=int, default=12)
    parser.add_argument('--distance_decay', type=float, default=1.0)
    return parser


def build_cooccurrence_graph(seq, num_nodes, distance_decay=1.0):
    adj_dict = defaultdict(lambda: defaultdict(float))
    for data in tqdm(seq, desc='Building graph'):
        seq_len = len(data)
        for i in range(seq_len):
            target = data[i]
            for k in range(1, 4):
                if i + k < seq_len:
                    neighbor = data[i + k]
                    decay_weight = distance_decay ** (k - 1)
                    adj_dict[target][neighbor] += decay_weight
                    adj_dict[neighbor][target] += decay_weight
    return adj_dict


def rank_graph(adj_dict, num_nodes, sample_num):
    adj = [[] for _ in range(num_nodes)]
    weight = [[] for _ in range(num_nodes)]

    for node_id in range(num_nodes):
        if len(adj_dict[node_id]) == 0:
            continue

        sorted_edges = sorted(adj_dict[node_id].items(), key=lambda x: x[1], reverse=True)
        top_k_edges = sorted_edges[:sample_num]
        adj[node_id] = [edge[0] for edge in top_k_edges]
        weight[node_id] = [edge[1] for edge in top_k_edges]

    return adj, weight


def main():
    opt = build_parser().parse_args()
    dataset_dir = resolve_dataset_dir(opt.dataset)
    sample_num = opt.sample_num

    print(
        f'Building Global Graph for {opt.dataset} | Sample Num: {sample_num} | '
        f'Distance Decay: {opt.distance_decay}'
    )

    seq = pickle.load(open(os.path.join(dataset_dir, 'all_train_seq.txt'), 'rb'))
    num_nodes = infer_num_nodes_from_sequences(seq)

    print('Calculating co-occurrences...')
    adj_dict = build_cooccurrence_graph(seq, num_nodes, distance_decay=opt.distance_decay)
    print('Filtering top-k edges...')
    adj, weight = rank_graph(adj_dict, num_nodes, sample_num)

    for node_id in range(num_nodes):
        pad_len = sample_num - len(adj[node_id])
        if pad_len > 0:
            adj[node_id].extend([0] * pad_len)
            weight[node_id].extend([0.0] * pad_len)

        adj[node_id] = adj[node_id][:sample_num]
        weight[node_id] = weight[node_id][:sample_num]

    print('Saving graphs...')
    pickle.dump(adj, open(os.path.join(dataset_dir, f'adj_{sample_num}.pkl'), 'wb'))
    pickle.dump(weight, open(os.path.join(dataset_dir, f'num_{sample_num}.pkl'), 'wb'))
    print('Done!')


if __name__ == '__main__':
    main()
