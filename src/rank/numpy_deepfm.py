"""DeepFM 的纯 numpy 推理实现（在线 Demo 用，替代 PyTorch 运行时）。

为什么需要这个模块
------------------
在线免费档（Streamlit Community Cloud）内存上限 1GB。实测各库的**固有导入开销**：

    裸 Python            9MB
    + numpy             25MB
    + scikit-learn     176MB
    + torch            643MB   ← 罪魁祸首
    + streamlit        676MB

也就是说，**只要 import torch，就固定花掉 643MB**，与数据规模无关。
把 micro 档数据从 100 万曝光继续压到 50 万曝光，峰值内存只从 847MB 降到 840MB——
瓶颈根本不是数据，而是框架运行时。

因此对**在线推理**这一条路径，把训练好的 DeepFM 权重导出为 numpy 数组，
用纯 numpy 复现前向计算，即可省下约 640MB，同时也不需要在线训练（省 20~120 秒）。

数学等价性
----------
本实现在数值上**严格等价于 `src/rank/deepfm.py:DeepFM.forward`**：
  - 一阶：`linear_dense(dense) + Σ linear_sparse[i](sparse[:,i])`
  - 二阶 FM：`0.5 * (||Σv||² - Σ||v||²)`
  - DNN：逐层 `Linear + ReLU`（推理时 Dropout 关闭）
  - 输出：三者相加的 logit（未过 sigmoid）
逐层用 numpy 矩阵乘法复现，无任何近似。

诚实边界
--------
本项目**不提出新算法**。DeepFM 仍是标准 DeepFM（Guo et al., 2017）；
本模块只是把**同一个模型**从 PyTorch 张量搬到 numpy 数组上跑，
目的是在受限内存的免费实例上演示，不改变任何方法语义。
完整档离线评估（`src/eval/evaluate.py`）仍使用 PyTorch 原版，权重与结论不受影响。
"""
from __future__ import annotations

import numpy as np


class NumpyDeepFM:
    """从 `DeepFMRanker` 导出的纯 numpy 前向实现（推理专用，不训练）。"""

    def __init__(self, params: dict):
        """
        参数
        ----
        params : dict
            由 `export_from_ranker()` 产出的权重字典：
              linear_dense_w (D,1), linear_dense_b (1,)
              linear_sparse_w (list of (C_i,1)), linear_sparse_b (list of (1,))
              sparse_embs (list of (C_i, E))
              dnn_weights (list of (out,in)), dnn_biases (list of (out,))
              dnn_acts (list of str)  —— 每层激活，'relu' 或 'linear'
        """
        self.ld_w = np.asarray(params["linear_dense_w"], dtype=np.float32)
        self.ld_b = np.asarray(params["linear_dense_b"], dtype=np.float32).reshape(1)
        self.ls_w = [np.asarray(w, dtype=np.float32) for w in params["linear_sparse_w"]]
        self.ls_b = [np.asarray(b, dtype=np.float32).reshape(1) for b in params["linear_sparse_b"]]
        self.embs = [np.asarray(e, dtype=np.float32) for e in params["sparse_embs"]]
        self.dw = [np.asarray(w, dtype=np.float32) for w in params["dnn_weights"]]
        self.db = [np.asarray(b, dtype=np.float32).reshape(-1) for b in params["dnn_biases"]]
        self.dact = list(params["dnn_acts"])
        self.n_sparse = len(self.embs)
        self.embed_dim = self.embs[0].shape[1] if self.n_sparse else 0

    # ------------------------------------------------------------------
    def forward_logit(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        """返回未过 sigmoid 的 logit，形状 (B,)。与 PyTorch 版逐一对应。"""
        dense = np.asarray(dense, dtype=np.float32)
        sparse = np.asarray(sparse, dtype=np.int64)
        if dense.ndim == 1:
            dense = dense[None, :]
        if sparse.ndim == 1:
            sparse = sparse[None, :]
        B = dense.shape[0]

        # ---- 一阶线性：dense 部分 ----
        logit = dense @ self.ld_w + self.ld_b            # (B,1)

        # ---- 一阶线性：sparse 部分（逐特征查表累加）----
        for i in range(self.n_sparse):
            logit = logit + self.ls_w[i][sparse[:, i]] + self.ls_b[i]

        # ---- 二阶 FM：0.5 * (||Σv||² - Σ||v||²) ----
        # 构造 (B, n_sparse, E)
        stacked = np.stack([self.embs[i][sparse[:, i]] for i in range(self.n_sparse)],
                           axis=1)                        # (B, S, E)
        sum_emb = stacked.sum(axis=1)                     # (B, E)
        sq_sum = (sum_emb * sum_emb).sum(axis=1)          # (B,)
        sum_sq = (stacked * stacked).sum(axis=(1, 2))     # (B,)
        fm = (0.5 * (sq_sum - sum_sq)).reshape(B, 1)      # (B,1)

        # ---- DNN：dense ⊕ 各稀疏 embedding 拼接，逐层 Linear(+ReLU) ----
        dnn_in = np.concatenate([dense] + [stacked[:, i, :] for i in range(self.n_sparse)],
                                axis=1)                   # (B, D + S*E)
        h = dnn_in
        for W, b, act in zip(self.dw, self.db, self.dact):
            h = h @ W.T + b
            if act == "relu":
                h = np.maximum(h, 0.0)
        dnn_out = h                                       # (B,1)

        return (logit + fm + dnn_out).reshape(-1)

    # ------------------------------------------------------------------
    def predict_ctr(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        """sigmoid(logit)，与 `DeepFMRanker.predict_ctr` 同口径。"""
        z = self.forward_logit(dense, sparse)
        return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------
def export_from_ranker(ranker) -> dict:
    """把 `DeepFMRanker`（PyTorch）的权重导出为 numpy 字典。

    仅在**离线/构建期**调用（此时 torch 已在内存中），
    导出结果可序列化为 .npz，供在线侧在**不导入 torch** 的前提下加载。
    """
    import torch  # 仅此处局部导入

    m = ranker.model
    with torch.no_grad():
        params = {
            "linear_dense_w": m.linear_dense.weight.detach().cpu().numpy().T,  # (D,1)
            "linear_dense_b": m.linear_dense.bias.detach().cpu().numpy(),
            "linear_sparse_w": [e.weight.detach().cpu().numpy() for e in m.linear_sparse],
            # 注意：`nn.Embedding(c, 1)` 无 bias（纯查表），二阶/一阶共用的 embedding 同理。
            # 用全零 bias 占位，保持 numpy 侧累加形式一致。
            "linear_sparse_b": [np.zeros(1, dtype=np.float32) for _ in m.linear_sparse],
            "sparse_embs": [e.weight.detach().cpu().numpy() for e in m.sparse_embs],
        }
        dw, db, acts = [], [], []
        for layer in m.dnn:
            if isinstance(layer, torch.nn.Linear):
                dw.append(layer.weight.detach().cpu().numpy())
                db.append(layer.bias.detach().cpu().numpy())
                acts.append("linear")
            elif isinstance(layer, torch.nn.ReLU):
                # 把激活挂到上一层（本层不带参数，直接改写最后一个标记）
                acts[-1] = "relu"
        params["dnn_weights"] = dw
        params["dnn_biases"] = db
        params["dnn_acts"] = acts
    return params


def save_npz(params: dict, path) -> None:
    """把导出参数存为 .npz。

    - 数值列表（每元素是 ndarray）→ 展平为 `key__0`, `key__1`, ... 并记 `key__len`
    - 字符串列表（如 dnn_acts）  → 整体存为一个 object 数组
    - 其余 ndarray               → 直接存
    """
    flat = {}
    for k, v in params.items():
        if isinstance(v, list):
            if v and isinstance(v[0], str):
                flat[k] = np.array(v, dtype=object)
            else:
                for i, arr in enumerate(v):
                    flat[f"{k}__{i}"] = np.asarray(arr)
                flat[f"{k}__len"] = np.array([len(v)])
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(path, **flat)


def load_npz(path) -> dict:
    """加载 .npz 还原参数字典（不需要 torch）。"""
    z = np.load(path, allow_pickle=True)
    out, handled = {}, set()
    for k in z.files:
        if k.endswith("__len"):
            base = k[: -len("__len")]
            if base in z.files:          # 字符串列表：整体存，不走展平路径
                continue
            n = int(z[k][0])
            out[base] = [z[f"{base}__{i}"] for i in range(n)]
            handled.update(f"{base}__{i}" for i in range(n))
            handled.add(k)
    for k in z.files:
        if k in handled or k.endswith("__len"):
            continue
        arr = z[k]
        if arr.dtype == object:                       # 字符串列表
            out[k] = [str(x) for x in arr.tolist()]
        else:
            out[k] = arr
    return out
