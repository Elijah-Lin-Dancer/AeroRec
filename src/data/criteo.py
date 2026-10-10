"""Criteo Display Advertising Challenge 真实 CTR 数据加载器 —— 路径 B 的「目标基准」。

Criteo 是 CTR / DeepFM 论文的**标准真实基准**，任务与本项目 DeepFM 排序器完全同构
（13 稠密 + 26 稀疏 -> p(click)）。本加载器与 ``src.data.adult.load_adult`` 同构：
只要产出 ``(dense, sparse, sparse_cards, y)`` 四元组，就能直接喂给同一个
``DeepFMRanker``，**模型零改动**。

格式（DAC ``train.txt``，每行 tab 分隔，无表头）：

    label \\t I1 \\t I2 ... I13 \\t C1 ... C26

  - label : 0 / 1（是否点击）
  - I1..I13 : 整数稠密特征，**缺失为空串**
  - C1..C26 : 十六进制哈希类别特征，**缺失为空串**

标准预处理（DeepFM / Criteo 胜出方案通用）：

  - dense : 缺失 -> 0，``log1p`` 压缩长尾，Z-score（仅用训练集统计量）
  - sparse: 按字段建词表，低频 / OOV -> 0，连续整数编码；每字段词表上限 ``max_card``

沙箱网络现状（已实测）：Criteo 官方（ailab.criteo.com / S3 criteo-labs / go.criteo.net）
全部 404，HuggingFace CDN 在本沙箱连接超时，故无法自动下载。本加载器已实现并**经单测
验证解析逻辑**；拿到 ``train.txt`` 后在本机一条命令即可跑出真实 Criteo AUC
（见 README「路径 B / Criteo 接入」）。

许可证：Criteo Display Advertising Challenge 数据集 CC-BY-NC-SA 4.0（研究 / 教学用途）。
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pandas as pd

N_DENSE = 13
N_SPARSE = 26
DEFAULT_MAX_CARD = 50_000          # 每稀疏字段词表上限（低频 / OOV 合并到码 0）
NAME_LABEL = "label"
DENSE_NAMES = [f"I{i}" for i in range(1, N_DENSE + 1)]
SPARSE_NAMES = [f"C{i}" for i in range(1, N_SPARSE + 1)]
COLUMNS = [NAME_LABEL] + DENSE_NAMES + SPARSE_NAMES

# 真实 CTR 基准的公开获取渠道（本沙箱均不可达，留作本机运行指引）：
#   - Kaggle:    "criteo-display-advertising-challenge"（需登录，含 train.txt + test.txt）
#   - 经典镜像:  dac.tar.gz（官方 S3 / go.criteo.net 现已 404，可自行检索可用镜像）
CRITEO_NOTE = (
    "Criteo 在构建沙箱网络被阻断（官方 S3 / go.criteo.net / HuggingFace 全部不可达）；"
    "本加载器已实现并经单测，拿到 train.txt 后本机 `python scripts/validate_real_data_deepfm.py "
    "--dataset criteo --criteo-path <train.txt>` 即可验证。"
)


def _read_lines(path: Path, n_rows: int | None):
    """返回行迭代器；若给定 n_rows 则只读前 n_rows 行（用于小样本快速验证）。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if n_rows is not None and i >= n_rows:
                break
            yield line.rstrip("\n")


def _build_vocab(lines, max_card: int) -> list[dict]:
    """第一遍：统计每稀疏字段词频，取 top-(max_card-1) 建码表（码 0 固定留给 OOV / 缺失）。"""
    counters = [collections.Counter() for _ in range(N_SPARSE)]
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 1 + N_DENSE + N_SPARSE:
            continue
        for j in range(N_SPARSE):
            tok = parts[1 + N_DENSE + j]
            if tok:
                counters[j][tok] += 1
    return [{tok: (i + 1) for i, (tok, _cnt) in enumerate(c.most_common(max_card - 1))}
            for c in counters]


def _encode(lines, vocab: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """第二遍：把行编码为 (dense_raw, sparse_codes, y)。

    dense_raw : (N, 13) float64，缺失填 0，**未**做 log / 标准化（留给上层）
    sparse_codes : (N, 26) int64，OOV / 缺失 -> 0
    y : (N,) int64
    """
    dense_rows, sparse_rows, ys = [], [], []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 1 + N_DENSE + N_SPARSE:
            continue
        dense = [0.0 if not parts[1 + k] else float(parts[1 + k]) for k in range(N_DENSE)]
        sparse = [vocab[j].get(parts[1 + N_DENSE + j], 0) for j in range(N_SPARSE)]
        dense_rows.append(dense)
        sparse_rows.append(sparse)
        ys.append(int(parts[0]))
    dense_arr = np.asarray(dense_rows, dtype=np.float64)
    sparse_arr = np.asarray(sparse_rows, dtype=np.int64)
    y_arr = np.asarray(ys, dtype=np.int64)
    return dense_arr, sparse_arr, y_arr


def _split(dense, sparse, y, val_split: float, seed: int):
    """分层切分训练 / 验证（保持验证集正类率与全量一致）。"""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_val_pos = int(round(len(pos) * val_split))
    n_val_neg = int(round(len(neg) * val_split))
    val_idx = np.concatenate([rng.choice(pos, n_val_pos, replace=False),
                              rng.choice(neg, n_val_neg, replace=False)])
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in range(len(y)) if i not in val_set])
    return train_idx, val_idx


def load_criteo(path: str | Path, n_rows: int | None = None,
                val_split: float = 0.1, max_card: int = DEFAULT_MAX_CARD,
                seed: int = 42) -> dict:
    """加载并预处理 Criteo DAC 数据，返回与 ``load_adult`` 同构的 dict。

    参数
    ----
    path     : train.txt 本地路径（本沙箱不可自动下载；请在本机放置）
    n_rows   : 仅读取前 N 行（默认 None = 全量；1e6 量级足够验证 DeepFM 迁移性）
    max_card : 每稀疏字段词表上限（控制嵌入表规模，低频 / OOV 合并到 0）

    返回 dict 字段（同 load_adult）：
        dense_train, sparse_train, y_train, dense_val, sparse_val, y_val,
        n_dense, sparse_cards, meta
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Criteo 文件不存在: {path}\n{CRITEO_NOTE}")

    lines_first = list(_read_lines(path, n_rows))
    vocab = _build_vocab(lines_first, max_card)
    dense, sparse, y = _encode(lines_first, vocab)

    if len(y) == 0:
        raise ValueError(f"Criteo 文件未解析出任何有效行: {path}")

    train_idx, val_idx = _split(dense, sparse, y, val_split, seed)
    dtr, dva = dense[train_idx], dense[val_idx]
    str_, sva = sparse[train_idx], sparse[val_idx]
    ytr, yva = y[train_idx], y[val_idx]

    # dense 预处理：log1p 压缩长尾 -> Z-score（仅用训练集统计量）
    dtr_t = np.log1p(np.clip(dtr, 0, None)).astype(np.float32)
    dva_t = np.log1p(np.clip(dva, 0, None)).astype(np.float32)
    mu, sd = dtr_t.mean(axis=0), dtr_t.std(axis=0) + 1e-8
    dtr_t = ((dtr_t - mu) / sd).astype(np.float32)
    dva_t = ((dva_t - mu) / sd).astype(np.float32)

    sparse_cards = [int(len(v) + 1) for v in vocab]   # +1 给 OOV / 缺失
    meta = {
        "dataset": "Criteo Display Advertising Challenge (2014)",
        "source": "Kaggle: criteo-display-advertising-challenge (CC-BY-NC-SA 4.0)",
        "license": "CC-BY-NC-SA 4.0 (research / teaching use)",
        "n_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "positive_rate_total": round(float(y.mean()), 4),
        "positive_rate_train": round(float(ytr.mean()), 4),
        "positive_rate_val": round(float(yva.mean()), 4),
        "n_dense": N_DENSE,
        "dense_cols": DENSE_NAMES,
        "sparse_cols": SPARSE_NAMES,
        "sparse_cards": sparse_cards,
        "max_card_per_field": int(max_card),
        "preprocessing": "dense: 缺失->0, log1p, Z-score(训练集); sparse: 缺失/OOV->0 的频率词表整数编码",
        "note": CRITEO_NOTE,
    }
    return {
        "dense_train": dtr_t, "sparse_train": str_, "y_train": ytr,
        "dense_val": dva_t, "sparse_val": sva, "y_val": yva,
        "n_dense": N_DENSE, "sparse_cards": sparse_cards, "meta": meta,
    }
