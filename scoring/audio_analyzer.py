"""
音频分析模块
负责VAD、重音检测、静音检测等音频特征提取
"""
import numpy as np
import librosa
import logging
from typing import List, Tuple, Optional
from .config import AUDIO_CONFIG, STRESS_CONFIG

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """音频分析器"""
    
    def __init__(self, sample_rate: int = None):
        """
        初始化音频分析器
        
        Args:
            sample_rate: 采样率，默认使用配置值
        """
        self.sample_rate = sample_rate or AUDIO_CONFIG["sample_rate"]
        self.vad_frame_duration = AUDIO_CONFIG["vad_frame_duration"]
        self.short_pause_threshold = AUDIO_CONFIG["short_pause_threshold"]
        self.long_pause_threshold = AUDIO_CONFIG["long_pause_threshold"]
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频数据（单声道，归一化到[-1, 1]）
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            # 归一化
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            return audio
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            raise
    
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        使用VAD检测语音段
        Args:
            audio: 音频数据
        Returns:
            语音段列表，每个元素为(start_time, end_time)
        """
        frame_length = int(self.sample_rate * self.vad_frame_duration)
        hop_length = frame_length // 2
        
        # 计算短时能量
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.mean(frame ** 2))
        
        if not energy:
            return []
        
        energy = np.array(energy)
        # 能量阈值（自适应）
        threshold = np.mean(energy) * 0.1
        
        # 检测语音段
        speech_segments = []
        in_speech = False
        start_time = 0
        
        for i, e in enumerate(energy):
            time = i * hop_length / self.sample_rate
            if e > threshold and not in_speech:
                in_speech = True
                start_time = time
            elif e <= threshold and in_speech:
                in_speech = False
                if time - start_time >= AUDIO_CONFIG["min_speech_duration"]:
                    speech_segments.append((start_time, time))
        
        # 处理最后一段
        if in_speech:
            end_time = len(audio) / self.sample_rate
            if end_time - start_time >= AUDIO_CONFIG["min_speech_duration"]:
                speech_segments.append((start_time, end_time))
        
        return speech_segments
    
    def detect_silences(self, audio: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        检测静音段
        
        Args:
            audio: 音频数据
            
        Returns:
            (short_pauses, long_pauses) - 短静音和长静音的时长列表
        """
        speech_segments = self.detect_speech_segments(audio)
        total_duration = len(audio) / self.sample_rate
        
        if not speech_segments:
            return [], []
        
        # 计算静音段
        silences = []
        prev_end = 0
        
        for start, end in speech_segments:
            if start > prev_end:
                silence_duration = start - prev_end
                silences.append(silence_duration)
            prev_end = end
        
        # 最后一段之后的静音
        if prev_end < total_duration:
            silence_duration = total_duration - prev_end
            silences.append(silence_duration)
        
        # 分类静音
        short_pauses = [s for s in silences 
                       if s >= self.short_pause_threshold and s < self.long_pause_threshold]
        long_pauses = [s for s in silences if s >= self.long_pause_threshold]
        
        return short_pauses, long_pauses
    
    def calculate_speech_chunks(self, audio: np.ndarray, text_words: List[str]) -> Tuple[List[int], float]:
        """
        计算语音块（chunks）信息
        
        Args:
            audio: 音频数据
            text_words: 文本单词列表
            
        Returns:
            (chunk_lengths, mean_deviation) - 语音块长度列表和平均偏差
        """
        speech_segments = self.detect_speech_segments(audio)
        
        if not speech_segments or not text_words:
            return [], 0.0
        
        # 估算每个语音段的单词数（简单按时间比例）
        total_speech_time = sum(end - start for start, end in speech_segments)
        total_words = len(text_words)
        
        if total_speech_time == 0:
            return [], 0.0
        
        # 计算每个语音段的单词数
        chunk_lengths = []
        for start, end in speech_segments:
            duration = end - start
            words_in_chunk = int((duration / total_speech_time) * total_words)
            if words_in_chunk > 0:
                chunk_lengths.append(words_in_chunk)
        
        if not chunk_lengths:
            return [], 0.0
        
        # 计算平均长度和平均绝对偏差
        mean_length = np.mean(chunk_lengths)
        mean_deviation = np.mean([abs(length - mean_length) for length in chunk_lengths])
        
        return chunk_lengths, mean_deviation
    
    def detect_stressed_syllables(self, audio: np.ndarray) -> List[float]:
        """
        检测重音音节位置
        
        Args:
            audio: 音频数据
            
        Returns:
            重音音节的时间戳列表
        """
        try:
            # 提取基频
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=STRESS_CONFIG["f0_range"][0],
                fmax=STRESS_CONFIG["f0_range"][1],
                frame_length=STRESS_CONFIG["frame_length"],
                hop_length=STRESS_CONFIG["hop_length"]
            )
            
            # 计算能量包络
            frame_length = STRESS_CONFIG["frame_length"]
            hop_length = STRESS_CONFIG["hop_length"]
            energy = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy.append(np.mean(frame ** 2))
            
            energy = np.array(energy)
            if len(energy) == 0:
                return []
            
            # 归一化能量
            if np.max(energy) > 0:
                energy = energy / np.max(energy)
            
            # 检测重音（能量高且基频变化大）
            stress_times = []
            threshold = STRESS_CONFIG["energy_threshold"]
            
            for i, e in enumerate(energy):
                if e > threshold:
                    time = i * hop_length / self.sample_rate
                    # 检查基频是否有效
                    if i < len(f0) and not np.isnan(f0[i]):
                        stress_times.append(time)
            
            return stress_times
            
        except Exception as e:
            logger.warning(f"重音检测失败，使用简化方法: {e}")
            # 简化方法：基于能量峰值
            return self._detect_stress_simple(audio)
    
    def _detect_stress_simple(self, audio: np.ndarray) -> List[float]:
        """简化的重音检测（仅基于能量）"""
        frame_length = STRESS_CONFIG["frame_length"]
        hop_length = STRESS_CONFIG["hop_length"]
        energy = []
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.mean(frame ** 2))
        
        if not energy:
            return []
        
        energy = np.array(energy)
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        threshold = STRESS_CONFIG["energy_threshold"]
        stress_times = []
        
        for i, e in enumerate(energy):
            if e > threshold:
                time = i * hop_length / self.sample_rate
                stress_times.append(time)
        
        return stress_times
    
    def calculate_stress_intervals(self, stress_times: List[float]) -> Tuple[float, float]:
        """
        计算重音间隔统计
        
        Args:
            stress_times: 重音时间戳列表
            
        Returns:
            (mean_interval, mean_deviation) - 平均间隔和平均偏差
        """
        if len(stress_times) < 2:
            return 0.0, 0.0
        
        intervals = [stress_times[i+1] - stress_times[i] 
                    for i in range(len(stress_times) - 1)]
        
        mean_interval = np.mean(intervals)
        mean_deviation = np.mean([abs(interval - mean_interval) for interval in intervals])
        
        return mean_interval, mean_deviation
    
    def get_audio_duration(self, audio: np.ndarray) -> float:
        """获取音频时长"""
        return len(audio) / self.sample_rate

