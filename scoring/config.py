"""
评分模块配置文件
基于TOEFL SpeechRater评分准则
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 评分权重配置（基于TOEFL SpeechRater研究）
DELIVERY_WEIGHTS = {
    "stretimemean": 0.15,      # 重音音节平均距离
    "wpsecutt": 0.15,          # 语速（每秒单词数）
    "wdpchk": 0.13,            # 平均语音块长度
    "wdpchkmeandev": 0.13,     # 语音块长度偏差
    "conftimeavg": 0.12,       # ASR置信度
    "repfreq": 0.08,           # 重复频率
    "silpwd": 0.06,            # 短静音频率（>0.15s）
    "ipc": 0.06,               # 中断点频率
    "stresyllmdev": 0.05,      # 重音音节距离偏差
    "L6": 0.03,                # 发音质量评分
    "longpfreq": 0.03,         # 长静音频率（>0.5s）
    "dpsec": 0.01,             # 不流畅词频率
}

LANGUAGE_WEIGHTS = {
    "types": 0.35,             # 词类型数量
    "poscvamax": 0.18,         # POS bigram对比
    "logfreq": 0.15,           # 词汇频率
    "lmscore": 0.11,           # 语言模型分数
    "tpsec": 0.11,             # 每秒词类型数
    "cvamax": 0.10,            # 单词数对比
}

# 音频分析参数
AUDIO_CONFIG = {
    "sample_rate": 16000,      # 采样率（与识别模块一致）
    "vad_frame_duration": 0.03,  # VAD帧长度（秒）
    "short_pause_threshold": 0.15,  # 短静音阈值（秒）
    "long_pause_threshold": 0.5,    # 长静音阈值（秒）
    "min_speech_duration": 0.1,    # 最小语音段长度（秒）
}

# 重音检测参数
STRESS_CONFIG = {
    "energy_threshold": 0.3,   # 能量阈值（归一化后）
    "f0_range": (80, 400),     # 基频范围（Hz）
    "frame_length": 2048,      # FFT帧长
    "hop_length": 512,         # 帧移
}

# 文本分析参数
TEXT_CONFIG = {
    "min_word_count": 10,      # 最小单词数
    "target_word_count_independent": 130,  # 独立任务目标单词数
    "target_word_count_integrated": 170,   # 综合任务目标单词数
    "disfluency_words": ["um", "uh", "er", "ah", "like", "you know"],  # 不流畅词
}

# 评分阈值
SCORE_CONFIG = {
    "max_score": 4.0,          # 满分
    "min_score": 0.0,          # 最低分
    "delivery_weight": 0.5,    # 发音部分权重
    "language_weight": 0.5,    # 内容部分权重
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(BASE_DIR, "logs", "scoring.log"),
}

# 模型路径（如果需要预训练模型）
MODEL_CONFIG = {
    "model_dir": os.path.join(BASE_DIR, "models"),
    # 可选：语言模型路径
    "lm_model_path": None,
}


# 讯飞星火语音评测配置
XUNFEI_CONFIG = {
    'APPID': '020a4aa7',
    'APISecret': 'MjI5MjM0MTI2ZGJlMWY3MTU3ZjcwODEx', 
    'APIKey': 'a5504a88c254edd71bd851db8c1ae78d',
    # 评测服务地址
    'HostUrl': 'wss://ise-api.xfyun.cn/v2/open-ise',
    'Sub': 'ise',
    'Ent': 'en_vip',
    'Category': 'topic',
}

# 综合评分权重配置
SCORING_WEIGHTS = {
    "local_algorithm": 0.4,   # 本地算法权重
    "large_model": 0.6,       # 大模型(讯飞)权重
}

