"""
音频工具模块
提供音频加载、保存、归一化等功能
"""
import numpy as np
import soundfile as sf
import librosa


class AudioProcessor:
    @staticmethod
    def load_6ch_audio(file_path, sr=16000):
        """加载6通道音频文件"""
        try:
            # 尝试多通道读取
            audio, sr = sf.read(file_path)
            if len(audio.shape) == 1:
                # 如果是单通道，复制为6通道
                audio = np.tile(audio.reshape(-1, 1), (1, 6))
            elif audio.shape[1] != 6:
                raise ValueError(f"音频通道数应为6，实际为{audio.shape[1]}")
            return audio.T, sr  # 返回形状为(6, samples)
        except Exception as e:
            print(f"加载音频失败: {e}")
            # 生成模拟的6通道测试音频
            return AudioProcessor.generate_test_audio(sr), sr
    
    @staticmethod
    def generate_test_audio(sr=16000, duration=5):
        """生成6通道测试音频（语音+噪声）"""
        t = np.linspace(0, duration, int(sr * duration))
        # 生成语音信号（正弦波模拟）
        speech = 0.5 * np.sin(2 * np.pi * 220 * t)
        speech = np.vstack([speech * (1 - i*0.1) for i in range(6)])
        
        # 添加不同方向的噪声
        noises = []
        for i in range(6):
            noise = 0.1 * np.random.randn(len(t))
            # 模拟空间差异
            delay = i * 3  # 样本延迟
            noise = np.roll(noise, delay)
            noises.append(noise)
        
        noise = np.array(noises)
        return speech + noise
    
    @staticmethod
    def save_audio(audio, file_path, sr=16000):
        """保存音频文件"""
        if len(audio.shape) > 1:
            audio = audio.T  # 转换为(samples, channels)
        sf.write(file_path, audio, sr)
    
    @staticmethod
    def normalize_audio(audio):
        """音频归一化"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio