"""Two-Tower 向量召回（PyTorch 双塔 + 负采样）。

思路
----
- 用户塔 / 物品塔各自把原始特征编码成同维向量；
- 用「点击对为正样本、随机负采样为负样本」训练，使点击过的 <用户,物品> 向量内积更大；
- 推理时用用户向量与全量物品向量做内积，取 Top-K，作为 ItemCF 的补充召回。

诚实边界：这是标准 Two-Tower 召回的实现（不提出新算法）；训练在 CPU 上对正样本
子采样进行，可复现。用户/物品特征仅含**原始特征**，不含生成器中的隐式交叉项。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import SEED, CATEGORIES, CITIES, TIME_SLOTS

# ---- 特征定义（用户级 / 物品级，与交互级特征库互补）----
USER_DENSE_COLS = [
    "aff_航线", "aff_酒店", "aff_商城", "aff_营销", "aff_直播",
    "price_sensitivity", "schedule_flexibility", "is_business",
    "active_days_7d", "miles_balance",
]
USER_SPARSE_COLS = ["home_idx"]
USER_SPARSE_CARDS = {"home_idx": len(CITIES)}

ITEM_DENSE_COLS = [
    "quality", "cash_price", "is_miles_ticket", "miles_required",
    "stock", "popularity",
]
ITEM_SPARSE_COLS = ["category_idx", "time_slot"]
ITEM_SPARSE_CARDS = {
    "category_idx": len(CATEGORIES),
    "time_slot": len(TIME_SLOTS) + 1,  # 0/1/2 早午晚 + 3 全天
}


def _standardize(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """按列做 z-score 标准化（std 下限 1e-6），返回 float32。"""
    out = []
    for c in cols:
        s = df[c].to_numpy(dtype=np.float32)
        mu = float(s.mean())
        std = float(s.std())
        if std < 1e-6:
            std = 1.0
        out.append((s - mu) / std)
    return np.stack(out, axis=1).astype(np.float32)


def build_user_features(users: pd.DataFrame):
    """用户级特征：返回 (dense, sparse)。"""
    dense = _standardize(users, USER_DENSE_COLS)
    sparse = users[USER_SPARSE_COLS].to_numpy(dtype=np.int64)
    return dense, sparse


def build_item_features(items: pd.DataFrame):
    """物品级特征：返回 (dense, sparse)。"""
    dense = _standardize(items, ITEM_DENSE_COLS)
    sparse = items[ITEM_SPARSE_COLS].to_numpy(dtype=np.int64)
    return dense, sparse


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple, out_dim: int):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoTowerModel(nn.Module):
    def __init__(self, user_dense_dim: int, item_dense_dim: int,
                 embed_dim: int = 64, hidden: tuple = (128, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_sparse_embs = nn.ModuleDict({
            k: nn.Embedding(card, embed_dim) for k, card in USER_SPARSE_CARDS.items()
        })
        self.item_sparse_embs = nn.ModuleDict({
            k: nn.Embedding(card, embed_dim) for k, card in ITEM_SPARSE_CARDS.items()
        })
        user_in = user_dense_dim + len(USER_SPARSE_COLS) * embed_dim
        item_in = item_dense_dim + len(ITEM_SPARSE_COLS) * embed_dim
        self.user_tower = MLP(user_in, hidden, embed_dim)
        self.item_tower = MLP(item_in, hidden, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _user_vec(self, u_dense, u_sparse):
        feats = [u_dense]
        for i, k in enumerate(USER_SPARSE_COLS):
            feats.append(self.user_sparse_embs[k](u_sparse[:, i]))
        x = torch.cat(feats, dim=-1)
        return nn.functional.normalize(self.user_tower(x), dim=-1)

    def _item_vec(self, i_dense, i_sparse):
        feats = [i_dense]
        for i, k in enumerate(ITEM_SPARSE_COLS):
            feats.append(self.item_sparse_embs[k](i_sparse[:, i]))
        x = torch.cat(feats, dim=-1)
        return nn.functional.normalize(self.item_tower(x), dim=-1)

    def forward(self, u_dense, u_sparse, i_dense, i_sparse):
        u = self._user_vec(u_dense, u_sparse)
        i = self._item_vec(i_dense, i_sparse)
        return (u * i).sum(dim=-1)  # 余弦相似度（向量已归一化）


class TwoTowerRecall:
    """Two-Tower 向量召回：训练后按用户向量做全量物品检索。"""

    def __init__(self, embed_dim: int = 64, hidden: tuple = (128, 64),
                 k_negatives: int = 4, epochs: int = 2, batch_size: int = 4096,
                 train_sample: int = 200_000, lr: float = 1e-3, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.k_negatives = k_negatives
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_sample = train_sample
        self.lr = lr
        self.device = device
        self.model = None
        self.user_vecs_ = None
        self.item_vecs_ = None
        self.user_ids_ = None
        self.item_ids_ = None

    def fit(self, logs: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        u_dense, u_sparse = build_user_features(users)
        i_dense, i_sparse = build_item_features(items)

        # 正样本：点击对（去重）
        pos = logs[logs["clicked"] == 1][["user_id", "item_id"]].drop_duplicates()
        if self.train_sample and len(pos) > self.train_sample:
            pos = pos.sample(n=self.train_sample, random_state=SEED)

        n_items = len(items)
        u_dense_t = torch.tensor(u_dense)
        u_sparse_t = torch.tensor(u_sparse)
        i_dense_t = torch.tensor(i_dense)
        i_sparse_t = torch.tensor(i_sparse)

        model = TwoTowerModel(
            user_dense_dim=u_dense.shape[1],
            item_dense_dim=i_dense.shape[1],
            embed_dim=self.embed_dim, hidden=self.hidden,
        ).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        pos_u = pos["user_id"].to_numpy()
        pos_i = pos["item_id"].to_numpy()
        n_pos = len(pos_u)

        model.train()
        for epoch in range(self.epochs):
            perm = np.random.permutation(n_pos)
            total_loss, n_batches = 0.0, 0
            for start in range(0, n_pos, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u_ids = pos_u[idx]
                i_ids = pos_i[idx]
                b = len(idx)

                # 负采样：每个正样本采样 K 个随机物品
                neg_ids = np.random.randint(0, n_items, size=(b, self.k_negatives))

                u_d_batch = u_dense_t[u_ids]
                u_s_batch = u_sparse_t[u_ids]
                pos_score = model(u_d_batch, u_s_batch,
                                  i_dense_t[i_ids], i_sparse_t[i_ids])  # [b]

                neg_d = i_dense_t[neg_ids].reshape(b * self.k_negatives, -1)
                neg_s = i_sparse_t[neg_ids].reshape(b * self.k_negatives, -1)
                u_d_neg = u_d_batch.repeat_interleave(self.k_negatives, dim=0)
                u_s_neg = u_s_batch.repeat_interleave(self.k_negatives, dim=0)
                neg_score = model(u_d_neg, u_s_neg, neg_d, neg_s).view(b, self.k_negatives)

                scores = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
                labels = torch.cat([
                    torch.ones(b, 1), torch.zeros(b, self.k_negatives)
                ], dim=1)

                loss = loss_fn(scores, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * b
                n_batches += b

            if n_batches:
                print(f"[two-tower] epoch {epoch + 1}/{self.epochs} "
                      f"avg_loss={total_loss / n_batches:.4f}")

        # 推理：计算全量用户/物品向量
        model.eval()
        self.user_vecs_ = self._embed_all(model._user_vec, u_dense_t, u_sparse_t)
        self.item_vecs_ = self._embed_all(model._item_vec, i_dense_t, i_sparse_t)
        self.user_ids_ = users["user_id"].to_numpy()
        self.item_ids_ = items["item_id"].to_numpy()
        self.model = model
        return self

    @staticmethod
    def _embed_all(fn, dense_t, sparse_t, chunk: int = 8192):
        outs = []
        with torch.no_grad():
            for s in range(0, len(dense_t), chunk):
                outs.append(fn(dense_t[s:s + chunk], sparse_t[s:s + chunk]))
        return torch.cat(outs, dim=0)

    def recall(self, user_id: int, k: int = 10) -> list:
        if self.item_vecs_ is None:
            raise RuntimeError("请先调用 fit() 训练")
        if user_id < 0 or user_id >= len(self.user_vecs_):
            return []
        uv = self.user_vecs_[user_id]                # [D]
        scores = self.item_vecs_ @ uv                # [N]
        top = torch.topk(scores, min(k, len(scores))).indices.cpu().numpy()
        return self.item_ids_[top].tolist()


if __name__ == "__main__":
    from src.data.synthetic_generator import load_dataset
    items, users, trips, logs = load_dataset()
    tt = TwoTowerRecall(epochs=1, train_sample=50_000).fit(logs, users, items)
    demo = int(users["user_id"].iloc[0])
    print("recall:", tt.recall(demo, k=10))
