# Session Splitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional long-session splitting with runtime-configurable threshold and fusion weight on top of the current `40.70` baseline.

**Architecture:** Keep the existing encoder and scoring path intact. Add a small pure helper for split-mask generation, then reuse the existing score computation for full/front/back views and blend them only for sessions longer than the configured threshold.

**Tech Stack:** Python, PyTorch runtime code, unittest for pure helper logic

---

### Task 1: Add failing tests for split-mask generation

**Files:**
- Create: `C:\Users\86155\Desktop\DMI-GNN-main\tests\test_session_split.py`
- Create: `C:\Users\86155\Desktop\DMI-GNN-main\session_split.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from session_split import build_split_masks


class SessionSplitTests(unittest.TestCase):
    def test_threshold_not_triggered_returns_original_masks(self):
        masks = [[1, 1, 1, 1, 0, 0]]
        front, back, triggered = build_split_masks(masks, split_threshold=8, front_ratio=0.6)
        self.assertEqual(front, masks)
        self.assertEqual(back, masks)
        self.assertEqual(triggered, [False])

    def test_long_session_splits_into_front_and_back(self):
        masks = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        front, back, triggered = build_split_masks(masks, split_threshold=8, front_ratio=0.6)
        self.assertEqual(front, [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(back, [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
        self.assertEqual(triggered, [True])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_session_split`
Expected: FAIL with `ModuleNotFoundError` for `session_split`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_session_split`
Expected: PASS

### Task 2: Add runtime controls and split-aware score fusion

**Files:**
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\main.py`
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\model.py`
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\session_split.py`

- [ ] **Step 1: Add parser arguments in `main.py`**

```python
parser.add_argument('--split_threshold', type=int, default=999, help='Only sessions longer than this trigger splitting')
parser.add_argument('--split_lambda', type=float, default=0.1, help='Fusion weight for split-session scores')
```

- [ ] **Step 2: Add tensor helper in `session_split.py`**

```python
import torch


def build_split_masks_tensor(mask, split_threshold, front_ratio=0.6):
    mask_cpu = mask.detach().cpu().tolist()
    front_cpu, back_cpu, triggered = build_split_masks(mask_cpu, split_threshold, front_ratio)
    device = mask.device
    front = torch.tensor(front_cpu, dtype=mask.dtype, device=device)
    back = torch.tensor(back_cpu, dtype=mask.dtype, device=device)
    triggered_tensor = torch.tensor(triggered, dtype=torch.bool, device=device)
    return front, back, triggered_tensor
```

- [ ] **Step 3: Refactor score computation in `model.py`**

```python
def _compute_scores_with_mask(self, hidden, mask):
    ...
    return max_scores, loss1 * self.beta, scores

def compute_scores(self, hidden, mask):
    full_scores, loss1, all_scores = self._compute_scores_with_mask(hidden, mask)
    if self.opt.split_lambda <= 0:
        return full_scores, loss1, all_scores
    ...
    return final_scores, loss1, all_scores
```

- [ ] **Step 4: Blend front/back scores only for triggered sessions**

```python
front_mask, back_mask, triggered = build_split_masks_tensor(mask, self.opt.split_threshold)
if triggered.any():
    front_scores, _, _ = self._compute_scores_with_mask(hidden, front_mask)
    back_scores, _, _ = self._compute_scores_with_mask(hidden, back_mask)
    split_scores = 0.5 * (front_scores + back_scores)
    mix = (1 - self.opt.split_lambda) * full_scores + self.opt.split_lambda * split_scores
    final_scores = torch.where(triggered.unsqueeze(-1), mix, full_scores)
else:
    final_scores = full_scores
```

- [ ] **Step 5: Run syntax validation**

Run: `python -m py_compile main.py model.py session_split.py`
Expected: PASS

### Task 3: Verify disabled and enabled behavior

**Files:**
- No new files

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest tests.test_session_split`
Expected: PASS

- [ ] **Step 2: Run broader local checks**

Run: `python -m unittest tests.test_build_graph tests.test_dataset_adaptation tests.test_preprocess_ecommerce tests.test_session_split`
Expected: PASS

- [ ] **Step 3: Smoke-test commands for later remote validation**

Disabled behavior:
```bash
python main.py --dataset Tmall --batch_size 100 --cl_rate 0.015 --temperature 0.07 --split_threshold 999 --split_lambda 0.1
```

Enabled behavior:
```bash
python main.py --dataset Tmall --batch_size 100 --cl_rate 0.015 --temperature 0.07 --split_threshold 15 --split_lambda 0.1
```
