"""
语音增强模块配置文件 (Speech Enhancement)
适配树莓派环境的音频处理配置
"""
import numpy as np
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # 音频参数
    SAMPLE_RATE = 16000
    CHANNELS = 1  # 默认1通道，适配USB麦克风（如果有多通道设备会自动检测）
    CHUNK_SIZE = 1024
    
    # 波束形成参数
    BEAMFORMER_TYPE = 'MVDR'  # 可选: 'DSB'(延迟求和), 'MVDR', 'GSC'
    LOOK_DIRECTION = 0  # 目标方向（弧度）
    MIC_POSITIONS = np.array([  # 假设麦克风为环形阵列
        [0.02, 0, 0],
        [0.01, 0.0173, 0],
        [-0.01, 0.0173, 0],
        [-0.02, 0, 0],
        [-0.01, -0.0173, 0],
        [0.01, -0.0173, 0]
    ])
    
    # 模型路径
    DENOISER_MODEL = "microsoft/asteroid_dprnntasnet-ks2_enh_v2"
    USE_PRETRAINED = True
    
    # 处理参数
    FRAME_LENGTH = 400  # 25ms
    HOP_LENGTH = 160    # 10ms
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = os.path.join(BASE_DIR, "logs", "recognition.log")