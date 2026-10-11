"""Criteo Display Advertising Challenge 真实 CTR 数据加载器 —— 路径 B 的「目标基准」。

Criteo 是 CTR / DeepFM 论文的**标准真实基准**，任务与本项目 DeepFM 排序器完全同构
（13 稠密 + 26 稀疏 -> p(click)）。本加载器与 ``src.data.adult.load_adult`` 同构：
只要产出 ``(dense, sparse, sparse_cards, y)`` 四元组，就能直接喂给同一个
``DeepFMRanker``，**模型零改动**。

支持两种输入格式（按扩展名自动分派）：

1. **DAC 文本**（``train.txt``，每行 tab 分隔，无表头）::

       label \t I1 \t I2 ... I13 \t C1 ... C26

   - label : 0 / 1（是否点击）
   - I1..I13 : 整数稠密特征，**缺失为空串**
   - C1..C26 : 十六进制哈希类别特征，**缺失为空串**

2. **Parquet**（如 HuggingFace ``eldieguinpo/criteo-1M`` 的 ``criteo_1M.parquet``）：
   列名即 ``label, I1..I13, C1..C26``；稠密缺失为 ``NaN``，稀疏类别为整数哈希码。

标准预处理（DeepFM / Criteo 胜出方案通用）：

  - dense : 缺失 -> 0，``log1p`` 压缩长尾，Z-score（仅用训练集统计量）
  - sparse: 按字段建词表，低频 / OOV -> 0，连续整数编码；每字段词表上限 ``max_card``

数据获取（本沙箱网络已实测：Criteo 官方 ailab / S3 criteo-labs / go.criteo.net 全部 404，
HuggingFace CDN 连接超时，故无法自动下载；请在本机获取后上传）：

  - HuggingFace ``eldieguinpo/criteo-1M`` -> ``criteo_1M.parquet``（约 45MB / 100 万行，推荐）
  - Kaggle ``criteo-display-advertising-challenge`` -> ``train.txt``（全量约 11GB）

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

CRITEO_NOTE = (
    "已用真实 criteo-1M（100 万行 parquet）跑通：DeepFM 验证 AUC 0.7733 / LR 基线 0.7717 / "
    "常数基线 0.5000（详见 data/real_validation/result.json）。"
    "注意：构建沙箱无法直接下载数据（官方 S3 / go.criteo.net / HuggingFace CDN 均不可达），"
    "需先在本机取得文件再放入 data/criteo/（该目录已 gitignore），"
    "然后运行 `python scripts/validate_real_data_deepfm.py --dataset criteo --criteo-path <file>`。"
)


# ---------------------------------------------------------------------------
# 文本格式（DAC train.txt）解析 —— 与既有单测契约保持一致
# ---------------------------------------------------------------------------
def _read_lines(path: Path, n_rows: int | None):
    """返回行迭代器；若给定 n_rows 则只读前 n_rows 行（用于小样本快速验证）。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if n_rows is not None and i >= n_rows:
                break
            yield line.rstrip("\n")


def _build_vocab(lines, max_card: int) -> list[dict]:
    """第一遍：统计每稀疏字段词频，取 top-(max_card-1) 建码表（码 0 固定留给 OOV / 缺失）。"""
    counters: list[collections.Counter] = [collections.Counter() for _ in range(N_SPARSE)]
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


# ---------------------------------------------------------------------------
# Parquet 格式解析（如 criteo-1M）
# ---------------------------------------------------------------------------
def _pick_cols(df: pd.DataFrame, names: list[str]) -> list[str]:
    """按（大小写不敏感的）列名从 DataFrame 中取出列，缺失则报错。"""
    lower = {str(c).lower(): c for c in df.columns}
    out = []
    for n in names:
        if n.lower() not in lower:
            raise ValueError(
                f"Criteo parquet 缺少列 '{n}'；现有列：{list(df.columns)[:8]}...（共 {len(df.columns)} 列）")
        out.append(lower[n.lower()])
    return out


def _build_vocab_frame(sparse_df: pd.DataFrame, max_card: int) -> list[dict]:
    """从 parquet 稀疏列建词表：每字段按值频取 top-(max_card-1)，码 0 留给 OOV / 缺失。"""
    vocab = []
    for col in sparse_df.columns:
        s = sparse_df[col]
        s = s[s.notna()]
        if s.dtype == object:                      # 文本 token：剔除空串
            s = s[s.astype(str).str.len() > 0]
        vc = s.value_counts()
        toks = list(vc.index[:max_card - 1])       # 索引即 token（int / str），直接作字典键
        vocab.append({tok: i + 1 for i, tok in enumerate(toks)})
    return vocab


def _encode_frame(sparse_df: pd.DataFrame, vocab: list[dict]) -> np.ndarray:
    """把 parquet 稀疏列按词表编码为 (N, 26) int64，OOV / 缺失 -> 0。"""
    codes = np.zeros((len(sparse_df), len(vocab)), dtype=np.int64)
    for j, col in enumerate(sparse_df.columns):
        codes[:, j] = sparse_df[col].map(vocab[j]).fillna(0).to_numpy(dtype=np.int64)
    return codes


def _load_parquet(path: Path, n_rows: int | None, max_card: int):
    """读 parquet，返回 (dense_raw, sparse_codes, y, sparse_cards)。"""
    df = pd.read_parquet(path)
    label_col = _pick_cols(df, [NAME_LABEL])[0]
    dense_cols = _pick_cols(df, DENSE_NAMES)
    sparse_cols = _pick_cols(df, SPARSE_NAMES)
    if n_rows is not None:
        df = df.iloc[:n_rows]

    y = df[label_col].to_numpy().astype(np.int64).ravel()
    dense_df = df[dense_cols].apply(pd.to_numeric, errors="coerce")
    dense_raw = np.nan_to_num(dense_df.to_numpy(dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
    sparse_df = df[sparse_cols]
    vocab = _build_vocab_frame(sparse_df, max_card)
    sparse_codes = _encode_frame(sparse_df, vocab)
    cards = [len(v) + 1 for v in vocab]
    return dense_raw, sparse_codes, y, cards


# ---------------------------------------------------------------------------
# 共享装配：切分 + 稠密预处理 + 元信息
# ---------------------------------------------------------------------------
def _split(y: np.ndarray, val_split: float, seed: int):
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


def _assemble(dense_raw, sparse_codes, y, sparse_cards, max_card, val_split, seed,
              source_meta: dict) -> dict:
    """把 (dense_raw, sparse_codes, y) 装配为与 load_adult 同构的返回 dict。"""
    if len(y) == 0:
        raise ValueError("Criteo 文件未解析出任何有效行")

    train_idx, val_idx = _split(y, val_split, seed)
    dtr, dva = dense_raw[train_idx], dense_raw[val_idx]
    str_, sva = sparse_codes[train_idx], sparse_codes[val_idx]
    ytr, yva = y[train_idx], y[val_idx]

    # dense 预处理：log1p 压缩长尾 -> Z-score（仅用训练集统计量）
    dtr_t = np.log1p(np.clip(dtr, 0, None)).astype(np.float32)
    dva_t = np.log1p(np.clip(dva, 0, None)).astype(np.float32)
    mu, sd = dtr_t.mean(axis=0), dtr_t.std(axis=0) + 1e-8
    dtr_t = ((dtr_t - mu) / sd).astype(np.float32)
    dva_t = ((dva_t - mu) / sd).astype(np.float32)

    meta = {
        **source_meta,
        "n_total": len(y),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "positive_rate_total": round(float(y.mean()), 4),
        "positive_rate_train": round(float(ytr.mean()), 4),
        "positive_rate_val": round(float(yva.mean()), 4),
        "n_dense": N_DENSE,
        "dense_cols": DENSE_NAMES,
        "sparse_cols": SPARSE_NAMES,
        "sparse_cards": [int(c) for c in sparse_cards],
        "max_card_per_field": int(max_card),
        "preprocessing": "dense: 缺失->0, log1p, Z-score(训练集); sparse: 缺失/OOV->0 的频率词表整数编码",
        "note": CRITEO_NOTE,
    }
    return {
        "dense_train": dtr_t, "sparse_train": str_, "y_train": ytr,
        "dense_val": dva_t, "sparse_val": sva, "y_val": yva,
        "n_dense": N_DENSE, "sparse_cards": [int(c) for c in sparse_cards], "meta": meta,
    }


_SOURCE_TEXT = {
    "dataset": "Criteo Display Advertising Challenge (2014)",
    "source": "DAC train.txt（Kaggle: criteo-display-advertising-challenge）",
    "license": "CC-BY-NC-SA 4.0 (research / teaching use)",
    "input_format": "dac_text",
}
_SOURCE_PARQUET = {
    "dataset": "Criteo Display Advertising Challenge (2014)",
    "source": "HuggingFace eldieguinpo/criteo-1M（Criteo DAC 均匀抽样，100 万行）",
    "license": "CC-BY-NC-SA 4.0 (research / teaching use)",
    "input_format": "parquet",
}


def load_criteo(path: str | Path, n_rows: int | None = None,
                val_split: float = 0.1, max_card: int = DEFAULT_MAX_CARD,
                seed: int = 42) -> dict:
    """加载并预处理 Criteo 数据（自动识别 DAC 文本 / parquet），返回与 ``load_adult`` 同构的 dict。

    参数
    ----
    path     : 本地文件路径。``*.parquet`` / ``*.pq`` 走 parquet 解析，其余按 DAC 文本解析。
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

    if path.suffix.lower() in (".parquet", ".pq"):
        dense_raw, sparse_codes, y, cards = _load_parquet(path, n_rows, max_card)
        source_meta = _SOURCE_PARQUET
    else:
        lines = list(_read_lines(path, n_rows))
        vocab = _build_vocab(lines, max_card)
        dense_raw, sparse_codes, y = _encode(lines, vocab)
        cards = [len(v) + 1 for v in vocab]
        source_meta = _SOURCE_TEXT

    return _assemble(dense_raw, sparse_codes, y, cards, max_card, val_split, seed, source_meta)
