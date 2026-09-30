"""召回层：热门兜底 + 协同过滤 + Two-Tower 向量召回 + 规则召回 + 融合。"""
from src.recall.baselines import PopularRecall, ItemCFRecall
from src.recall.two_tower import TwoTowerRecall
from src.recall.rule import RuleRecall
from src.recall.fusion import RecallFusion

__all__ = [
    "PopularRecall", "ItemCFRecall", "TwoTowerRecall",
    "RuleRecall", "RecallFusion",
]
