"""
基于自监督学习的深度学习评分器 (SSL Scorer)
使用 Wav2Vec 2.0 / HuBERT 提取特征进行端到端评分
"""
import torch
import torch.nn as nn
import logging
import os
import numpy as np
import librosa
from typing import Optional, Dict

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入transformers (需要用户安装)
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("未安装transformers库，无法使用SSL评分器")

class PronunciationRegressionHead(nn.Module):
    """简单的回归头，将Wav2Vec特征映射到分数"""
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid() # 输出归一化到0-1，后续再缩放
        )
        
    def forward(self, x):
        return self.layers(x)

class SSLScorer:
    """基于自监督学习(Self-Supervised Learning)模型的评分器"""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请先安装 transformers 和 torch: pip install transformers torch")
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"正在加载SSL模型: {model_name} (设备: {self.device})...")
        
        try:
            # 加载预训练的Wav2Vec2
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.encoder = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.encoder.eval() # 冻结编码器，只做特征提取
            
            # 加载评分头 (这里演示用随机初始化的，实际需要加载训练好的权重)
            # 在实际项目中，这里应该加载一个训练好的 .pth 文件
            self.regressor = PronunciationRegressionHead().to(self.device)
            self.regressor.eval()
            
            # 尝试加载本地微调过的权重（如果有）
            local_weights = os.path.join(os.path.dirname(__file__), "models", "ssl_scorer.pth")
            if os.path.exists(local_weights):
                logger.info(f"加载本地评分模型权重: {local_weights}")
                self.regressor.load_state_dict(torch.load(local_weights, map_location=self.device))
            else:
                logger.warning("未找到本地评分模型权重，将使用随机初始化参数进行演示！")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def predict_score(self, audio_path: str) -> float:
        """
        预测发音分数
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            predicted_score: 预测分数 (0-4分)
        """
        try:
            # 1. 加载和预处理音频
            # Wav2Vec2 需要 16kHz 音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 2. 准备输入
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            
            # 3. 提取特征 (不计算梯度)
            with torch.no_grad():
                outputs = self.encoder(input_values)
                # 取最后一层隐藏状态: [Batch, Time, Dim] -> [1, T, 768]
                hidden_states = outputs.last_hidden_state
                
                # 4. 池化 (Mean Pooling) -> [1, 768]
                pooled_features = torch.mean(hidden_states, dim=1)
                
                # 5. 回归预测
                normalized_score = self.regressor(pooled_features).item()
                
            # 映射回 0-4 分
            final_score = normalized_score * 4.0
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"深度学习评分失败: {e}")
            return 0.0

if __name__ == "__main__":
    # 简单测试
    try:
        scorer = SSLScorer()
        # 创建个假音频测试
        dummy_audio = "temp_test.wav"
        import soundfile as sf
        sf.write(dummy_audio, np.random.uniform(-1, 1, 16000*3), 16000)
        
        score = scorer.predict_score(dummy_audio)
        print(f"测试音频得分: {score}/4.0")
        
        os.remove(dummy_audio)
    except Exception as e:
        print(f"测试失败: {e}")
