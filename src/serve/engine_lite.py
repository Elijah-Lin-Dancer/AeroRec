        return np.asarray(self._d[key])


class _ABItems(_ABUsers):
    pass


class _IlocProxy:
    """按整数数组取行，返回支持 `p[列]` / `p[列].to_numpy()` / `.reset_index()` 的对象。"""

    def __init__(self, d: dict):
        self._d = d

    def __getitem__(self, idx):
        return _RowProxy({k: np.asarray(v)[idx] for k, v in self._d.items()})


class _RowProxy:
    def __init__(self, d: dict):
        self._d = d

    def __getitem__(self, k):
        if isinstance(k, list):                      # u[[col1, col2]] → 需要 .to_numpy()
            return _ColsProxy([self._d[c] for c in k])
        return _ColProxy(self._d[k])

    def reset_index(self, drop: bool = False):
        return self                                  # numpy 无索引，原样返回

    def to_numpy(self):
        return np.stack(list(self._d.values()), axis=1)


class _ColsProxy:
    """多列包装：支持 `.to_numpy()`，把多列按列堆叠成 (n, k) 数组。"""

    def __init__(self, arrays: list):
        self._arrays = [np.asarray(a) for a in arrays]

    def to_numpy(self, dtype=None):
        out = np.stack(self._arrays, axis=1)
        return out.astype(dtype) if dtype is not None else out


class _ColProxy:
    """列包装：`arr.to_numpy()` 与直接当 ndarray 用都支持。"""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def to_numpy(self, dtype=None):
        return self._arr.astype(dtype) if dtype is not None else self._arr

    def __array__(self, dtype=None, copy=None):
        return self._arr.astype(dtype) if dtype is not None else self._arr


if __name__ == "__main__":
    import resource
    import time

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    t = time.time()
    eng = LiteRecommendationEngine()
    print(f"[engine] init {time.time() - t:.1f}s peak_rss={rss():.0f}MB")
    t = time.time()
    res = eng.recommend(0, 10)
    print(f"[recommend] {time.time() - t:.3f}s -> {len(res)} items")
    print("  top3:", [d["name"] for d in res[:3]])
    t = time.time()
    ab = eng.run_ab(n_users=300)
    print(f"[ab] {time.time() - t:.1f}s  CTR {ab['ctrl_ctr']:.4f}->{ab['trt_ctr']:.4f} "
          f"({ab['ctr_lift']:+.1%})  CVR {ab['ctrl_cvr']:.4f}->{ab['trt_cvr']:.4f} "
          f"({ab['cvr_lift']:+.1%})")
    print(f"[FINAL] peak_rss={rss():.0f}MB")
