"""DeepFM 排序模型（PyTorch）：FM（一阶 + 二阶交叉）+ DNN。

诚实边界：这是标准 DeepFM（Guo et al., 2017）的实现，不提出新算法。
"""
import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, dense_dim: int, sparse_cards: list[int],
                 embed_dim: int = 16, dnn_hidden: tuple = (64, 32),
                 dropout: float = 0.1):
        super().__init__()
        self.n_sparse = len(sparse_cards)

        # 一阶线性
        self.linear_dense = nn.Linear(dense_dim, 1)
        self.linear_sparse = nn.ModuleList([nn.Embedding(c, 1) for c in sparse_cards])

        # 二阶 FM 与 DNN 共享的稀疏 embedding
        self.sparse_embs = nn.ModuleList([nn.Embedding(c, embed_dim) for c in sparse_cards])

        # DNN
        dnn_in = dense_dim + self.n_sparse * embed_dim
        layers = []
        prev = dnn_in
        for h in dnn_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.dnn = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, dense, sparse):
        # 一阶线性
        logit = self.linear_dense(dense)                     # [B, 1]
        for i in range(self.n_sparse):
            logit = logit + self.linear_sparse[i](sparse[:, i])

        # 二阶 FM 交叉：0.5 * (||sum v||^2 - sum ||v||^2)
        embs = torch.stack([self.sparse_embs[i](sparse[:, i]) for i in range(self.n_sparse)],
                           dim=1)                            # [B, n_sparse, E]
        sum_emb = embs.sum(dim=1)                            # [B, E]
        sq_sum = (sum_emb * sum_emb).sum(dim=1)              # [B]
        sum_sq = (embs * embs).sum(dim=(1, 2))               # [B]
        fm = 0.5 * (sq_sum - sum_sq).unsqueeze(1)            # [B, 1]

        # DNN
        dnn_in = torch.cat([dense] + [embs[:, i, :] for i in range(self.n_sparse)], dim=1)
        dnn_out = self.dnn(dnn_in)

        return logit + fm + dnn_out  # logit（未过 sigmoid）
