"""
评分模块主类
整合所有子模块，提供统一的评分接口
"""
import logging
import os
from typing import Optional, Dict
from .audio_analyzer import AudioAnalyzer
from .delivery_scorer import DeliveryScorer
from .language_scorer import LanguageScorer
from .score_calculator import ScoreCalculator
from .feedback_generator import FeedbackGenerator
from .config import LOG_CONFIG

# 尝试导入讯飞评分器
try:
    from .xunfei_rater import XunfeiRater
except ImportError:
    XunfeiRater = None

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class ScoreResult:
    """评分结果类"""
    
    def __init__(self, raw_score: float, delivery_score: float, language_score: float,
                 delivery_features: Dict, language_features: Dict,
                 feedback_en: str, feedback_zh: str):
        self.raw_score = raw_score
        self.delivery_score = delivery_score
        self.language_score = language_score
        self.delivery_features = delivery_features
        self.language_features = language_features
        self.feedback_en = feedback_en
        self.feedback_zh = feedback_zh
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "raw_score": self.raw_score,
            "delivery_score": self.delivery_score,
            "language_score": self.language_score,
            "delivery_features": self.delivery_features,
            "language_features": self.language_features,
            "feedback_en": self.feedback_en,
            "feedback_zh": self.feedback_zh,
        }


class SpeechRater:
    """语音评分器主类"""
    
    def __init__(self):
        """初始化评分器"""
        logger.info("初始化语音评分器...")
        
        self.audio_analyzer = AudioAnalyzer()
        self.delivery_scorer = DeliveryScorer()
        self.language_scorer = LanguageScorer()
        self.score_calculator = ScoreCalculator()
        self.feedback_generator = FeedbackGenerator()
        
        # 初始化讯飞评分器
        if XunfeiRater:
            self.xunfei_rater = XunfeiRater()
        else:
            self.xunfei_rater = None
        
        logger.info("语音评分器初始化完成")
    
    def score_with_xunfei(self, audio_path: str, text: str) -> Optional[Dict]:
        """使用讯飞大模型进行评分"""
        if not self.xunfei_rater:
            logger.warning("讯飞评分器未初始化")
            return None
        return self.xunfei_rater.score(audio_path, text)

    def score(self, audio_path: str, text: str, 
              asr_confidence: Optional[float] = None,
              task_type: str = "independent") -> ScoreResult:
        """
        评分主接口
        
        Args:
            audio_path: 音频文件路径（必需）
            text: 识别得到的文本（必需）
            asr_confidence: ASR平均置信度（可选）
            task_type: 任务类型（"independent" 或 "integrated"）
            
        Returns:
            ScoreResult对象
        """
        try:
            logger.info(f"开始评分: 音频={audio_path}, 文本长度={len(text)}")
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 获取音频时长
            audio = self.audio_analyzer.load_audio(audio_path)
            audio_duration = self.audio_analyzer.get_audio_duration(audio)
            
            # 1. 计算发音特征
            logger.debug("计算发音特征...")
            delivery_features_raw = self.delivery_scorer.calculate_all_features(
                audio_path, text, asr_confidence
            )
            delivery_features_norm = self.delivery_scorer.normalize_features(delivery_features_raw)
            delivery_score = self.delivery_scorer.calculate_delivery_score(delivery_features_norm)
            
            # 2. 计算内容特征
            logger.debug("计算内容特征...")
            language_features_raw = self.language_scorer.calculate_all_features(
                text, audio_duration, task_type
            )
            language_features_norm = self.language_scorer.normalize_features(language_features_raw)
            language_score = self.language_scorer.calculate_language_score(language_features_norm)
            
            # 3. 计算最终分数
            logger.debug("计算最终分数...")
            final_score = self.score_calculator.calculate_final_score(
                delivery_score, language_score
            )
            
            # 4. 生成评价
            logger.debug("生成评价...")
            score_result_dict = self.score_calculator.get_score_breakdown(
                delivery_score, language_score,
                delivery_features_raw, language_features_raw
            )
            feedback = self.feedback_generator.generate_feedback(score_result_dict)
            
            # 5. 构建结果对象
            result = ScoreResult(
                raw_score=final_score,
                delivery_score=delivery_score * 4,  # 转换为0-4分
                language_score=language_score * 4,
                delivery_features=delivery_features_raw,
                language_features=language_features_raw,
                feedback_en=feedback["en"],
                feedback_zh=feedback["zh"]
            )
            
            logger.info(f"评分完成: 总分={final_score:.2f}, 发音={delivery_score*4:.2f}, 内容={language_score*4:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"评分过程出错: {e}", exc_info=True)
            raise

