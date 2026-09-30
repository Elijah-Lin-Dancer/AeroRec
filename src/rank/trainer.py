"""CTR 排序模型训练器（DeepFM）。"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import SEED
from src.rank.deepfm import DeepFM


class DeepFMRanker:
    def __init__(self, dense_dim: int, sparse_cards: list[int],
                 embed_dim: int = 16, dnn_hidden: tuple = (64, 32),
                 lr: float = 1e-3, epochs: int = 2, batch_size: int = 2048,
                 device: str = "cpu"):
        self.dense_dim = dense_dim
        self.sparse_cards = sparse_cards
        self.model = DeepFM(dense_dim, sparse_cards, embed_dim, dnn_hidden).to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, dense, sparse, y, val_dense=None, val_sparse=None, val_y=None):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        dense_t = torch.tensor(dense)
        sparse_t = torch.tensor(sparse, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(y_t)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total, cnt = 0.0, 0
            for s in range(0, n, self.batch_size):
                bidx = perm[s:s + self.batch_size]
                logit = self.model(dense_t[bidx], sparse_t[bidx])
                loss = loss_fn(logit, y_t[bidx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(bidx)
                cnt += len(bidx)
            msg = f"[deepfm] epoch {epoch + 1}/{self.epochs} loss={total / cnt:.4f}"
            if val_dense is not None:
                msg += f" val_auc={self._auc(val_dense, val_sparse, val_y):.4f}"
            print(msg)
        self.model.eval()
        return self

    def _auc(self, dense, sparse, y):
        return roc_auc_score(y, self.predict_ctr(dense, sparse))

    def predict_ctr(self, dense, sparse) -> np.ndarray:
        self.model.eval()
        dense_t = torch.tensor(dense)
        sparse_t = torch.tensor(sparse, dtype=torch.long)
        outs = []
        with torch.no_grad():
            for s in range(0, len(dense_t), 8192):
                logit = self.model(dense_t[s:s + 8192], sparse_t[s:s + 8192])
                outs.append(torch.sigmoid(logit).squeeze(-1))
        return torch.cat(outs).cpu().numpy()
