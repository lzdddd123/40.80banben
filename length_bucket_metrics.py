BUCKET_ORDER = ('SHORT', 'MEDIUM', 'LONG')


def bucket_name_for_length(length):
    if length <= 3:
        return 'SHORT'
    if length <= 7:
        return 'MEDIUM'
    return 'LONG'


def summarize_length_buckets(lengths, hits, mrrs):
    total = len(lengths)
    summary = {
        name: {'count': 0, 'hit_sum': 0.0, 'mrr_sum': 0.0, 'ratio': 0.0, 'hr': 0.0, 'mrr': 0.0}
        for name in BUCKET_ORDER
    }

    for length, hit_value, mrr_value in zip(lengths, hits, mrrs):
        bucket = bucket_name_for_length(length)
        summary[bucket]['count'] += 1
        summary[bucket]['hit_sum'] += float(hit_value)
        summary[bucket]['mrr_sum'] += float(mrr_value)

    for bucket in BUCKET_ORDER:
        count = summary[bucket]['count']
        if total > 0:
            summary[bucket]['ratio'] = count * 100.0 / total
        if count > 0:
            summary[bucket]['hr'] = summary[bucket]['hit_sum'] * 100.0 / count
            summary[bucket]['mrr'] = summary[bucket]['mrr_sum'] * 100.0 / count

    return summary


def format_length_bucket_report(summary):
    lines = [
        '',
        '=' * 50,
        '[Ablation Analysis] Session Length Buckets',
        '=' * 50,
    ]
    for bucket in BUCKET_ORDER:
        item = summary[bucket]
        lines.append(f'[{bucket}] (占比: {item["ratio"]:.1f}%, N={item["count"]}):')
        lines.append(f'   -> HR@20:  {item["hr"]:.4f}')
        lines.append(f'   -> MRR@20: {item["mrr"]:.4f}')
    lines.append('=' * 50)
    return '\n'.join(lines)
