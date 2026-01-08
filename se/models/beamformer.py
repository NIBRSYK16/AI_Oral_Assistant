"""
波束形成器模块
实现延迟求和波束形成和MVDR波束形成算法
"""
import numpy as np
import scipy.signal as signal
from scipy.linalg import solve


class Beamformer:
    def __init__(self, mic_positions, fs=16000, direction=0):
        """
        初始化波束形成器
        
        参数:
            mic_positions: 麦克风位置数组 (M, 3)
            fs: 采样率
            direction: 目标方向（弧度）
        """
        self.mic_positions = mic_positions
        self.num_mics = mic_positions.shape[0]
        self.fs = fs
        self.direction = direction
        
        # 声速 (m/s)
        self.c = 343.0
        
        # 计算目标方向的延迟
        self.delays = self._calculate_delays()
        
        # 滤波器长度
        self.filter_length = 256
    
    def _calculate_delays(self):
        """计算各麦克风相对于参考点的延迟"""
        # 假设声源在远场
        # 方向向量
        direction_vec = np.array([
            np.cos(self.direction),
            np.sin(self.direction),
            0
        ])
        
        # 计算各麦克风的时延（以采样点为单位）
        delays = []
        for pos in self.mic_positions:
            # 投影到方向向量
            proj = np.dot(pos, direction_vec)
            # 转换为时间延迟（秒）
            time_delay = proj / self.c
            # 转换为采样点延迟
            sample_delay = time_delay * self.fs
            delays.append(sample_delay)
        
        # 调整为相对于第一个麦克风的相对延迟
        delays = np.array(delays) - delays[0]
        return delays
    
    def delay_and_sum(self, multi_channel_audio):
        """
        延迟求和波束形成
        
        参数:
            multi_channel_audio: 多通道音频 (channels, samples)
            
        返回:
            增强后的单通道音频
        """
        num_channels, num_samples = multi_channel_audio.shape
        enhanced = np.zeros(num_samples)
        
        for ch in range(num_channels):
            # 计算延迟（四舍五入到最近的整数）
            delay = int(round(self.delays[ch]))
            
            if delay >= 0:
                # 延迟该通道
                delayed_signal = np.concatenate([
                    np.zeros(delay),
                    multi_channel_audio[ch, :-delay] if delay > 0 else multi_channel_audio[ch]
                ])
                # 确保长度一致
                delayed_signal = delayed_signal[:num_samples]
            else:
                # 提前该通道
                delayed_signal = multi_channel_audio[ch, -delay:]
                delayed_signal = np.concatenate([
                    delayed_signal,
                    np.zeros(-delay)
                ])
            
            # 求和
            enhanced += delayed_signal
        
        # 平均
        enhanced = enhanced / num_channels
        return enhanced
    
    def mvdr_beamformer(self, multi_channel_audio, noise_covariance=None):
        """
        MVDR波束形成器
        简化版本，适合实时处理
        """
        # 如果未提供噪声协方差矩阵，使用对角矩阵
        if noise_covariance is None:
            noise_covariance = np.eye(self.num_mics)
        
        # 计算导向向量
        steering_vector = np.exp(-1j * 2 * np.pi * np.arange(512)[:, None] * 
                                self.delays[None, :] / self.fs)
        
        # 计算MVDR权重
        try:
            # 简化计算
            R_inv = np.linalg.inv(noise_covariance + 1e-6 * np.eye(self.num_mics))
            w = (R_inv @ steering_vector.T.conj()) / \
                (steering_vector @ R_inv @ steering_vector.T.conj())
            
            # 应用权重
            enhanced = np.sum(w[:, :, None] * multi_channel_audio, axis=1)
            return enhanced.real
        except:
            # 如果计算失败，回退到延迟求和
            print("MVDR计算失败，使用延迟求和")
            return self.delay_and_sum(multi_channel_audio)