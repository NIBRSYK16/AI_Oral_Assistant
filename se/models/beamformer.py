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
    
    def directional_enhancement(self, multi_channel_audio, method='gcc_phat'):
        """
        基于语音方位的针对性音频增强
        自动检测语音来源方向，然后使用自适应波束形成增强该方向的音频
        
        参数:
            multi_channel_audio: 多通道音频 (channels, samples)
            method: DOA估计方法，可选 'gcc_phat' (默认) 或 'srp_phat'
            
        返回:
            增强后的单通道音频
        """
        try:
            # 1. 估计语音来源方向 (DOA - Direction of Arrival)
            estimated_angle = self._estimate_doa(multi_channel_audio, method=method)
            
            # 2. 根据估计的方向计算导向向量和延迟
            direction_vec = np.array([
                np.cos(estimated_angle),
                np.sin(estimated_angle),
                0
            ])
            
            # 计算各麦克风的时延
            delays = []
            for pos in self.mic_positions:
                proj = np.dot(pos, direction_vec)
                time_delay = proj / self.c
                sample_delay = time_delay * self.fs
                delays.append(sample_delay)
            
            delays = np.array(delays) - delays[0]  # 相对延迟
            
            # 3. 使用自适应MVDR波束形成增强该方向的音频
            num_channels, num_samples = multi_channel_audio.shape
            
            # 估计噪声协方差矩阵（使用前几帧作为噪声估计）
            noise_frames = min(10, num_samples // 100)
            if noise_frames > 0:
                noise_samples = multi_channel_audio[:, :noise_frames * (self.fs // 100)]
                noise_cov = np.cov(noise_samples)
            else:
                noise_cov = np.eye(num_channels) * 0.01
            
            # 计算导向向量（频域）
            n_fft = 512
            freqs = np.fft.fftfreq(n_fft, 1.0 / self.fs)
            freqs = freqs[:n_fft // 2 + 1]
            
            steering_vector = np.zeros((len(freqs), num_channels), dtype=complex)
            for i, f in enumerate(freqs):
                if f > 0:
                    phase_delays = -2j * np.pi * f * delays
                    steering_vector[i, :] = np.exp(phase_delays)
            
            # 计算MVDR权重
            try:
                R_inv = np.linalg.inv(noise_cov + 1e-6 * np.eye(num_channels))
                
                # 分段处理音频
                hop_length = n_fft // 2
                enhanced_frames = []
                
                for start_idx in range(0, num_samples, hop_length):
                    end_idx = min(start_idx + n_fft, num_samples)
                    frame = multi_channel_audio[:, start_idx:end_idx]
                    
                    if frame.shape[1] < n_fft:
                        # 补零
                        padding = n_fft - frame.shape[1]
                        frame = np.pad(frame, ((0, 0), (0, padding)), mode='constant')
                    
                    # FFT
                    frame_fft = np.fft.rfft(frame, n=n_fft, axis=1)
                    
                    # 对每个频率应用MVDR权重
                    enhanced_fft = np.zeros_like(frame_fft[0, :])
                    for i in range(len(freqs)):
                        if freqs[i] > 0:
                            w = (R_inv @ steering_vector[i, :].conj()) / \
                                (steering_vector[i, :] @ R_inv @ steering_vector[i, :].conj() + 1e-10)
                            enhanced_fft[i] = np.sum(w * frame_fft[:, i])
                    
                    # IFFT
                    enhanced_frame = np.fft.irfft(enhanced_fft, n=n_fft)[:end_idx - start_idx]
                    enhanced_frames.append(enhanced_frame)
                
                enhanced = np.concatenate(enhanced_frames)
                enhanced = enhanced[:num_samples]  # 确保长度一致
                
                return enhanced.real
                
            except Exception as e:
                # 如果MVDR计算失败，使用延迟求和作为回退
                print(f"自适应波束形成计算失败: {e}，使用延迟求和")
                # 临时更新方向并计算延迟
                original_direction = self.direction
                self.direction = estimated_angle
                self.delays = delays
                enhanced = self.delay_and_sum(multi_channel_audio)
                self.direction = original_direction
                self.delays = self._calculate_delays()
                return enhanced
                
        except Exception as e:
            print(f"方向性增强失败: {e}，使用延迟求和")
            return self.delay_and_sum(multi_channel_audio)
    
    def _estimate_doa(self, multi_channel_audio, method='gcc_phat'):
        """
        估计语音来源方向 (DOA)
        
        参数:
            multi_channel_audio: 多通道音频 (channels, samples)
            method: 估计方法，'gcc_phat' 或 'srp_phat'
            
        返回:
            估计的角度（弧度）
        """
        num_channels, num_samples = multi_channel_audio.shape
        
        if method == 'gcc_phat':
            # 使用GCC-PHAT方法估计时延差
            # 选择第一个麦克风作为参考
            ref_channel = 0
            max_correlation = -np.inf
            best_angle = self.direction  # 默认使用初始方向
            
            # 在可能的角度范围内搜索
            angles_to_test = np.linspace(-np.pi, np.pi, 72)  # 5度步进
            
            for angle in angles_to_test:
                # 计算该角度下的理论时延
                direction_vec = np.array([np.cos(angle), np.sin(angle), 0])
                delays = []
                for pos in self.mic_positions:
                    proj = np.dot(pos, direction_vec)
                    time_delay = proj / self.c
                    delays.append(time_delay)
                delays = np.array(delays) - delays[ref_channel]
                
                # 计算GCC-PHAT相关性
                correlation = 0.0
                for ch in range(1, num_channels):
                    # 计算互相关
                    ref_sig = multi_channel_audio[ref_channel, :]
                    ch_sig = multi_channel_audio[ch, :]
                    
                    # 简化的时延相关（使用互相关）
                    # 这里使用简单的互相关作为近似
                    if delays[ch] != 0:
                        delay_samples = int(round(delays[ch] * self.fs))
                        if abs(delay_samples) < len(ref_sig) // 2:
                            if delay_samples > 0:
                                shifted_ref = np.pad(ref_sig[delay_samples:], (delay_samples, 0), mode='constant')
                            else:
                                shifted_ref = np.pad(ref_sig[:delay_samples], (0, -delay_samples), mode='constant')
                            
                            # 计算相关性
                            corr = np.corrcoef(shifted_ref[:len(ch_sig)], ch_sig)[0, 1]
                            if not np.isnan(corr):
                                correlation += abs(corr)
                
                if correlation > max_correlation:
                    max_correlation = correlation
                    best_angle = angle
            
            return best_angle
            
        else:  # srp_phat (简化版)
            # SRP-PHAT方法（简化实现）
            # 类似于GCC-PHAT，但在所有麦克风对上进行
            best_angle = self.direction
            max_energy = -np.inf
            
            angles_to_test = np.linspace(-np.pi, np.pi, 72)
            
            for angle in angles_to_test:
                direction_vec = np.array([np.cos(angle), np.sin(angle), 0])
                delays = []
                for pos in self.mic_positions:
                    proj = np.dot(pos, direction_vec)
                    time_delay = proj / self.c
                    delays.append(time_delay)
                delays = np.array(delays) - delays[0]
                
                # 计算该方向下的能量（延迟对齐后）
                total_energy = 0.0
                for ch in range(num_channels):
                    delay_samples = int(round(delays[ch] * self.fs))
                    if abs(delay_samples) < num_samples // 2:
                        if delay_samples > 0:
                            shifted = np.pad(multi_channel_audio[ch, delay_samples:], 
                                           (delay_samples, 0), mode='constant')
                        else:
                            shifted = np.pad(multi_channel_audio[ch, :delay_samples], 
                                           (0, -delay_samples), mode='constant')
                        total_energy += np.sum(shifted[:num_samples] ** 2)
                
                if total_energy > max_energy:
                    max_energy = total_energy
                    best_angle = angle
            
            return best_angle