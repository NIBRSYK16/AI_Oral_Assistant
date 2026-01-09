"""
分数计算模块
整合发音和内容评分，计算最终分数
"""
import logging
from typing import Dict
from .config import SCORE_CONFIG

logger = logging.getLogger(__name__)


class ScoreCalculator:
    """分数计算器"""
    
    def __init__(self):
        """初始化分数计算器"""
        self.max_score = SCORE_CONFIG["max_score"]
        self.min_score = SCORE_CONFIG["min_score"]
        self.delivery_weight = SCORE_CONFIG["delivery_weight"]
        self.language_weight = SCORE_CONFIG["language_weight"]
    
    def calculate_final_score(self, delivery_score: float, language_score: float) -> float:
        """
        计算最终分数
        
        Args:
            delivery_score: 发音分数（0-1）
            language_score: 内容分数（0-1）
            
        Returns:
            最终分数（1.5-4）
        """
        # 加权求和 (得到 0-1)
        combined_score = (delivery_score * self.delivery_weight + 
                         language_score * self.language_weight)
        
        # 映射到 1.5 - 4 分
        # Formula: 1.5 + score * 2.5
        final_score = 1.5 + (combined_score * 2.5)
        
        # 确保在有效范围内
        final_score = max(1.5, min(self.max_score, final_score))
        
        return round(final_score, 2)
    
    def get_score_breakdown(self, delivery_score: float, language_score: float,
                          delivery_features: Dict, language_features: Dict) -> Dict:
        """
        获取分数详情
        
        Args:
            delivery_score: 发音分数
            language_score: 内容分数
            delivery_features: 发音特征详情
            language_features: 内容特征详情
            
        Returns:
            分数详情字典
        """
        final_score = self.calculate_final_score(delivery_score, language_score)
        
        return {
            "final_score": final_score,
            "delivery_score": round(1.5 + (delivery_score * 2.5), 2),  # 转换为1.5-4分
            "language_score": round(1.5 + (language_score * 2.5), 2),
            "delivery_features": delivery_features,
            "language_features": language_features,
            "max_score": self.max_score,
        }

