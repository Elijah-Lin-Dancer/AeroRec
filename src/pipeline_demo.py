"""Demo：生成数据 → 训练召回基线 → 为样例用户产出推荐。

运行：python -m src.pipeline_demo
"""
from src.config import TOP_K
from src.data.synthetic_generator import build_dataset
from src.recall.baselines import PopularRecall, ItemCFRecall


def main():
    items, users, logs = build_dataset()
    print(f"[data] items={len(items)} users={len(users)} "
          f"logs={len(logs)} overall_ctr={logs['clicked'].mean():.4f}")

    # 大日志量下采样子集训练召回，控制耗时
    train_logs = logs
    if len(logs) > 1_000_000:
        train_logs = logs.sample(n=1_000_000, random_state=42)
    pop = PopularRecall(top_k=200).fit(train_logs)
    icf = ItemCFRecall(top_k=200).fit(train_logs)

    # 选一个有点击行为的样例用户做演示
    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    uid = clicked_by_user.index[0]
    user_clicks = clicked_by_user.loc[uid]

    pop_rec = pop.recall(k=TOP_K)
    icf_rec = icf.recall(list(user_clicks), k=TOP_K)

    name = dict(zip(items["item_id"], items["name"]))
    cat = dict(zip(items["item_id"], items["category"]))

    print(f"\n[demo] user_id={uid}  历史点击物品数={len(user_clicks)}")
    print("\n--- 热门兜底召回 Top{} ---".format(TOP_K))
    for i in pop_rec:
        print(f"  {name[i]:>12}  [{cat[i]}]")
    print("\n--- ItemCF 个性化召回 Top{} ---".format(TOP_K))
    for i in icf_rec:
        print(f"  {name[i]:>12}  [{cat[i]}]")

    print("\n[ok] 召回层基线跑通：候选集就绪，下游排序层(W2)将继续精排。")


if __name__ == "__main__":
    main()
