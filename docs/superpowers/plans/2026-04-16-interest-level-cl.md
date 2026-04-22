# Interest-Level CL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-risk interest-level contrastive loss on top of the existing Tmall-favorable training pipeline without changing inference behavior.

**Architecture:** Keep the current graph construction, aggregators, routing, and scoring path unchanged at inference time. Reuse the existing two-view training setup and projector, but add a second contrastive term that aligns the most active per-interest representations across the two stochastic views.

**Tech Stack:** Python, PyTorch

---

### Task 1: Add interest-level CL controls

**Files:**
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\main.py`

- [ ] Add `interest_cl_rate` and `interest_cl_topk` parser arguments.
- [ ] Keep defaults conservative so old behavior can be recovered by setting `interest_cl_rate=0.0`.

### Task 2: Expose per-interest features from the existing routing path

**Files:**
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\model.py`

- [ ] Factor the existing interest extraction logic into a reusable helper.
- [ ] Return per-interest representations, route strengths, and the existing interest regularization term from that helper.
- [ ] Keep `compute_scores()` output unchanged for evaluation compatibility.

### Task 3: Add interest-level contrastive loss

**Files:**
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\model.py`

- [ ] Implement `interest_ssl_loss()` using the top-k most active interests selected from the two-view route strengths.
- [ ] Reuse the existing projector and temperature for a minimal, low-risk extension.

### Task 4: Wire the new loss into training only

**Files:**
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\main.py`

- [ ] Compute the new interest-level CL only when `interest_cl_rate > 0`.
- [ ] Add it to the existing training loss without modifying evaluation logic.

### Task 5: Verify syntax and interface stability

**Files:**
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\main.py`
- Modify: `C:\Users\86155\Desktop\DMI-GNN-main\model.py`

- [ ] Run `python -m py_compile main.py model.py`.
- [ ] Confirm the training script still accepts the old command line and only adds optional flags.
