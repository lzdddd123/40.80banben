import os


def _iter_dataset_roots(datasets_root=None):
    if datasets_root is not None:
        yield datasets_root
        return

    benchmark_root = os.path.expanduser('~/autodl-tmp/benchmark_datasets')
    local_root = os.path.join(os.getcwd(), 'datasets')

    seen = set()
    for root in (benchmark_root, local_root):
        normalized = os.path.normpath(root)
        if normalized not in seen:
            seen.add(normalized)
            yield root


def resolve_dataset_dir(dataset, datasets_root=None):
    checked_roots = []
    lowered = dataset.lower()

    for root in _iter_dataset_roots(datasets_root):
        checked_roots.append(root)
        if not os.path.isdir(root):
            continue

        exact_path = os.path.join(root, dataset)
        if os.path.isdir(exact_path):
            return exact_path

        for entry in os.listdir(root):
            candidate = os.path.join(root, entry)
            if os.path.isdir(candidate) and entry.lower() == lowered:
                return candidate

    joined_roots = ', '.join(checked_roots)
    raise FileNotFoundError(f'Dataset directory for {dataset} not found under {joined_roots}')


def infer_num_nodes_from_sequences(sequences):
    max_item = 0
    for seq in sequences:
        if seq:
            max_item = max(max_item, max(seq))
    return max_item + 1
