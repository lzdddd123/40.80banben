try:
    import torch
except ImportError:  # pragma: no cover - local env fallback
    torch = None


def build_split_masks(masks, split_threshold, front_ratio=0.6):
    front_masks, back_masks, triggered = [], [], []
    for mask in masks:
        valid_len = int(sum(mask))
        if valid_len <= split_threshold:
            front_masks.append(list(mask))
            back_masks.append(list(mask))
            triggered.append(False)
            continue

        cut = int(valid_len * front_ratio)
        cut = max(1, min(valid_len - 1, cut))

        front = [0] * len(mask)
        back = [0] * len(mask)
        for idx in range(cut):
            front[idx] = 1
        for idx in range(cut, valid_len):
            back[idx] = 1

        front_masks.append(front)
        back_masks.append(back)
        triggered.append(True)
    return front_masks, back_masks, triggered


def build_split_masks_tensor(mask, split_threshold, front_ratio=0.6):
    if torch is None:
        raise RuntimeError('torch is required for tensor-based split masks')

    mask_cpu = mask.detach().cpu().tolist()
    front_cpu, back_cpu, triggered = build_split_masks(mask_cpu, split_threshold, front_ratio)
    device = mask.device
    front = torch.tensor(front_cpu, dtype=mask.dtype, device=device)
    back = torch.tensor(back_cpu, dtype=mask.dtype, device=device)
    triggered_tensor = torch.tensor(triggered, dtype=torch.bool, device=device)
    return front, back, triggered_tensor


def fuse_split_scores(full_scores, split_scores, triggered, split_lambda):
    fused = []
    for base_row, split_row, is_triggered in zip(full_scores, split_scores, triggered):
        if is_triggered:
            fused.append([
                (1 - split_lambda) * base + split_lambda * split_value
                for base, split_value in zip(base_row, split_row)
            ])
        else:
            fused.append(list(base_row))
    return fused


def fuse_split_scores_tensor(full_scores, split_scores, triggered, split_lambda):
    if torch is None:
        raise RuntimeError('torch is required for tensor-based score fusion')

    if split_lambda <= 0:
        return full_scores
    if not torch.any(triggered):
        return full_scores

    mixed_scores = (1 - split_lambda) * full_scores + split_lambda * split_scores
    return torch.where(triggered.unsqueeze(-1), mixed_scores, full_scores)
