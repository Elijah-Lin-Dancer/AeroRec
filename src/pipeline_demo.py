"""Demo：数据 → 四路召回（热门/ItemCF/Two-Tower/规则）→ 融合 → 样例推荐。

运行：python -m src.pipeline_demo
说明：Two-Tower 为演示速度采用较小训练规模，正式复现可调大。
"""
from src.config import TOP_K
from src.data.synthetic_generator import build_dataset
from src.recall.baselines import PopularRecall, ItemCFRecall
from src.recall.two_tower import TwoTowerRecall
from src.recall.rule import RuleRecall
from src.recall.fusion import RecallFusion


def main():
    items, users, trips, logs = build_dataset()
    print(f"[data] items={len(items)} users={len(users)} trips={len(trips)} "
          f"logs={len(logs)} overall_ctr={logs['clicked'].mean():.4f}")

    # 召回基线的训练日志：大日志量下采样子集控制耗时
    train_logs = logs if len(logs) <= 1_000_000 else logs.sample(n=1_000_000, random_state=42)

    print("\n[recall] 训练四路召回 ...")
    pop = PopularRecall(top_k=200).fit(train_logs)
    icf = ItemCFRecall(top_k=200).fit(train_logs)
    tt = TwoTowerRecall(epochs=1, train_sample=50_000).fit(logs, users, items)
    rule = RuleRecall().fit(logs=logs, items=items, users=users, trips=trips)

    # 选一个有点击行为的样例用户
    clicked_by_user = logs[logs["clicked"] == 1].groupby("user_id")["item_id"].apply(list)
    uid = int(clicked_by_user.index[0])
    user_clicks = clicked_by_user.loc[uid]

    # 四路融合（权重：规则 > 向量 > 协同 > 热门）
    fusion = RecallFusion(max_candidates=200, top_per_source=100)
    fusion.add("popular", 0.6, lambda k: pop.recall(k=k))
    fusion.add("itemcf", 1.0, lambda k: icf.recall(list(user_clicks), k=k))
    fusion.add("two_tower", 1.2, lambda k: tt.recall(uid, k=k))
    fusion.add("rule", 1.5, lambda k: rule.recall(uid, k=k))
    fused = fusion.recall()

    name = dict(zip(items["item_id"], items["name"]))
    cat = dict(zip(items["item_id"], items["category"]))

    print(f"\n[demo] user_id={uid}  历史点击物品数={len(user_clicks)}")
    for label, recs in [
        ("热门兜底", pop.recall(k=TOP_K)),
        ("ItemCF", icf.recall(list(user_clicks), k=TOP_K)),
        ("Two-Tower", tt.recall(uid, k=TOP_K)),
        ("规则召回", rule.recall(uid, k=TOP_K)),
        ("融合候选 Top10", fused[:TOP_K]),
    ]:
        print(f"\n--- {label} ---")
        for i in recs:
            print(f"  {name[i]:>12}  [{cat[i]}]")

    print(f"\n[ok] 四路召回 + 融合跑通：融合候选集 {len(fused)} 个，将交给排序层精排。")


if __name__ == "__main__":
    main()
