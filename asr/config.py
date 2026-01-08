"""
语音识别模块配置文件
适配树莓派环境的ASR配置
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ASRConfig:
    """ASR配置类"""
    
    # 音频参数（必须与语音增强模块一致）
    SAMPLE_RATE = 16000  # Vosk推荐16kHz
    CHANNELS = 1  # 单声道
    
    # Vosk模型配置
    # 树莓派4推荐使用vosk-model-small-en-us-0.15（约40MB）
    # 如果需要更高精度，可以使用vosk-model-en-us-0.22（约1.8GB，但树莓派可能较慢）
    VOSK_MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15")
    VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    
    # 如果模型不存在，是否自动下载
    AUTO_DOWNLOAD_MODEL = False  # 树莓派上建议手动下载
    
    # 识别参数
    MAX_ALTERNATIVES = 3  # 最大候选结果数
    WORDS = True  # 是否返回词级时间戳
    PARTIAL_WORDS = False  # 是否返回部分词
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = os.path.join(BASE_DIR, "logs", "asr.log")
