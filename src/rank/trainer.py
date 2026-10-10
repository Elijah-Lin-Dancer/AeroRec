"""CTR 排序模型训练器（DeepFM）。"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import SEED
from src.rank.deepfm import DeepFM

# ---------------------------------------------------------------------------
# DeepFM 训练预算：锚定「梯度更新次数」，而不是固定 epoch 数。
#
# 背景（实测教训）：模型能否收敛取决于**优化器步数**，而这个量级与数据规模无关。
# 若只按 epoch 计数，数据量小的档位会被严重欠训 —— 轻量档曾用
# train_sample=12,000 / epochs=2 / batch=2048，仅 12 次梯度更新，
# 实测 test AUC = 0.5109（≈随机），推荐效果反而**劣于热门基线**。
# 同一份数据的对照实验：
#     12 次 → AUC 0.5109      60 次 → 0.6991
#    120 次 → AUC 0.7422     480 次 → 0.8008（与完整档 0.8019 同级）
#
# 因此各档位统一「锚定目标更新次数 → 反推 epoch」；batch 取 512：
# 同样样本量下更新次数是 2048 的 4 倍，且 batch 更小、峰值内存更低。
# ---------------------------------------------------------------------------
TARGET_UPDATES = 400
DEEFM_BATCH_SIZE = 512


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def plan_epochs(train_sample: int, batch_size: int = DEEFM_BATCH_SIZE,
                target_updates: int = TARGET_UPDATES, min_epochs: int = 2) -> int:
    """由目标梯度更新次数反推 epoch 数。

    用法：`epochs = plan_epochs(len(y))`，保证训练量不随数据规模缩小而坍塌。
    """
    batches = max(1, ceil_div(train_sample, batch_size))
    return max(int(min_epochs), ceil_div(target_updates, batches))


class DeepFMRanker:
    def __init__(self, dense_dim: int, sparse_cards: list[int],
                 embed_dim: int = 16, dnn_hidden: tuple = (64, 32),
                 lr: float = 1e-3, epochs: int = 2, batch_size: int = 2048,
                 device: str = "cpu"):
        self.dense_dim = dense_dim
        self.sparse_cards = sparse_cards
        # 权重初始化必须发生在全局随机种子被设定之后。
        # DeepFM.__init__ 内部会做 xavier_uniform_ / normal_(std=0.01)，
        # 若此处不先播种，网络初值将取决于进程此前消耗掉的 RNG 状态，
        # 导致同一份代码在不同依赖版本/不同调用顺序下产出不同的指标。
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.model = DeepFM(dense_dim, sparse_cards, embed_dim, dnn_hidden).to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, dense, sparse, y, val_dense=None, val_sparse=None, val_y=None):
        # 训练阶段的随机性是显式的：打乱顺序（randperm）与 Dropout 掩码都依赖全局 RNG。
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
