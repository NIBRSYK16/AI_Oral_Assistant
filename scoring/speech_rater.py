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
from .config import LOG_CONFIG, SCORING_WEIGHTS

# 尝试导入讯飞评分器
try:
    from .xunfei_rater import XunfeiRater
except ImportError:
    XunfeiRater = None

# 配置日志
logger = logging.getLogger(__name__)

# 配置日志文件输出
try:
    log_file = LOG_CONFIG["file"]
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
    
    # 将处理器添加到 scoring 包的根 logger，这样 xunfei_rater 等子模块的日志也会被记录
    package_logger = logging.getLogger("scoring")
    package_logger.addHandler(file_handler)
    package_logger.setLevel(getattr(logging, LOG_CONFIG["level"]))
    
    # 确保 logger 变量指向当前模块的 logger (用于本文件的日志记录)
    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"Warning: Failed to configure file logging: {e}")
    # 确保 logger 变量已定义
    logger = logging.getLogger(__name__)

# 保留基本配置以防独立运行
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG["level"]),
        format=LOG_CONFIG["format"]
    )


class ScoreResult:
    """评分结果类"""
    
    def __init__(self, raw_score: float, delivery_score: float, language_score: float,
                 delivery_features: Dict, language_features: Dict,
                 feedback_en: str, feedback_zh: str,
                 recognized_text: Optional[str] = None):
        self.raw_score = raw_score
        self.delivery_score = delivery_score
        self.language_score = language_score
        self.delivery_features = delivery_features
        self.language_features = language_features
        self.feedback_en = feedback_en
        self.feedback_zh = feedback_zh
        self.recognized_text = recognized_text
    
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
            "recognized_text": self.recognized_text
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
              task_type: str = "independent",
              reference_text: Optional[str] = None) -> ScoreResult:
        """
        评分主接口
        
        Args:
            audio_path: 音频文件路径（必需）
            text: 识别得到的文本（用于内容分析）
            asr_confidence: ASR平均置信度（可选）
            task_type: 任务类型（"independent" 或 "integrated"）
            reference_text: 参考文本或题目（用于讯飞评测topic模式或朗读模式）
            
        Returns:
            ScoreResult对象
        """
        try:
            logger.info(f"开始评分: 音频={audio_path}, 文本长度={len(text)}")
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 0. 尝试调用讯飞评分 (如果有)
            xunfei_result = None
            if self.xunfei_rater:
                # 优先使用reference_text(题目)，否则使用text
                eval_text = reference_text if reference_text else text
                logger.info(f"调用讯飞评分, 文本: {eval_text[:20]}...")
                xunfei_result = self.xunfei_rater.score(audio_path, eval_text)
            
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
            
            # 3. 计算最终分数 (本地算法)
            logger.debug("计算最终分数...")
            local_score = self.score_calculator.calculate_final_score(
                delivery_score, language_score
            )
            final_score = local_score
            
            # 使用讯飞结果进行加权融合 (如果存在)
            if xunfei_result and 'total_score' in xunfei_result:
                # 1. 获取讯飞分数 (转换到1.5-4分制)
                if 'converted_score' in xunfei_result:
                    xunfei_score = xunfei_result['converted_score']
                else:
                    xunfei_score = 1.5 + (xunfei_result['total_score'] / 5.0 * 2.5)
                
                # 2. 尝试融合维度分数 (如果有细分维度)
                # 注意：delivery_score, language_score 此时还是 0-1 的归一化值，需要保持在这里处理，
                # 最后输出时才统一转 1.5-4
                if 'fluency_score' in xunfei_result:
                    xf_delivery = xunfei_result['fluency_score'] / 5.0 
                    # 发音分 (保持0-1) = 本地 * 0.4 + 讯飞 * 0.6
                    delivery_score = delivery_score * SCORING_WEIGHTS["local_algorithm"] + xf_delivery * SCORING_WEIGHTS["large_model"]
                    
                if 'accuracy_score' in xunfei_result:
                    xf_language = xunfei_result['accuracy_score'] / 5.0
                    language_score = language_score * SCORING_WEIGHTS["local_algorithm"] + xf_language * SCORING_WEIGHTS["large_model"]
                
                # 3. 计算加权总分 (local_score 已经是 1.5-4 分了，需要在 score_calculator 里面改过了)
                # 实际上 calculate_final_score 已经被改为返回 1.5-4
                # 所以这里 xunfei_score 也是 1.5-4
                final_score = (local_score * SCORING_WEIGHTS["local_algorithm"] + 
                              xunfei_score * SCORING_WEIGHTS["large_model"])
                
                logger.info(f"融合评分: 本地={local_score:.2f}, 讯飞={xunfei_score:.2f}, 最终={final_score:.2f}")
            
            # 4. 生成评价
            logger.debug("生成评价...")
            score_result_dict = self.score_calculator.get_score_breakdown(
                delivery_score, language_score,
                delivery_features_raw, language_features_raw
            )
            feedback = self.feedback_generator.generate_feedback(score_result_dict)
            
            # 5. 构建结果对象
            # 尝试提取讯飞返回的识别文本
            recognized_text = xunfei_result.get("recognized_text") if xunfei_result else None

            result = ScoreResult(
                raw_score=final_score,
                delivery_score=1.5 + (delivery_score * 2.5),  # 转换为1.5-4分
                language_score=1.5 + (language_score * 2.5),
                delivery_features=delivery_features_raw,
                language_features=language_features_raw,
                feedback_en=feedback["en"],
                feedback_zh=feedback["zh"],
                recognized_text=recognized_text
            )
            
            logger.info(f"评分完成: 总分={final_score:.2f}, 发音={1.5 + delivery_score*2.5:.2f}, 内容={1.5 + language_score*2.5:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"评分过程出错: {e}", exc_info=True)
            raise

