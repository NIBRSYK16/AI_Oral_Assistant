"""
唤醒词检测模块
使用轻量级方法实现持续监听和唤醒词检测
"""
import numpy as np
import pyaudio
import time
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """唤醒词检测器 - 轻量级实现"""
    
    def __init__(self, 
                 wake_keywords: list = None,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 energy_threshold: float = 0.01,
                 silence_duration: float = 0.5):
        """
        初始化唤醒词检测器
        
        Args:
            wake_keywords: 唤醒关键词列表，默认 ["assistant", "voice assistant"]
            sample_rate: 采样率
            chunk_size: 音频块大小
            energy_threshold: 能量阈值，低于此值认为是静音
            silence_duration: 静音持续时间（秒），用于判断语音结束
        """
        self.wake_keywords = wake_keywords or ["assistant", "voice assistant", "语音助手"]
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        
        # 音频参数
        self.format = pyaudio.paInt16
        self.channels = 1  # 单声道用于唤醒检测
        
        # PyAudio实例
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # 控制标志
        self.is_listening = False
        self._stop_flag = threading.Event()
        
        # 回调函数
        self.on_wake_detected: Optional[Callable[[str], None]] = None
        
        # 音频缓冲区（用于简单关键词检测）
        self.audio_buffer = []
        self.max_buffer_duration = 3.0  # 最多缓存3秒音频
        
        logger.info(f"唤醒词检测器初始化完成，唤醒词: {self.wake_keywords}")
    
    def set_wake_callback(self, callback: Callable[[str], None]):
        """
        设置唤醒回调函数
        
        Args:
            callback: 检测到唤醒词时调用的函数，参数为识别到的文本
        """
        self.on_wake_detected = callback
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """计算音频能量"""
        if len(audio_data) == 0:
            return 0.0
        return np.mean(audio_data ** 2)
    
    def _is_speech(self, audio_data: np.ndarray) -> bool:
        """判断是否为语音（基于能量）"""
        energy = self._calculate_energy(audio_data)
        return energy > self.energy_threshold
    
    def _simple_keyword_match(self, audio_data: np.ndarray) -> Optional[str]:
        """
        简单的关键词匹配（基于能量模式）
        这是一个简化实现，实际应该使用ASR模型
        
        Returns:
            如果检测到唤醒词，返回匹配的关键词，否则返回None
        """
        # 这里使用简单的能量检测作为占位符
        # 实际应该：
        # 1. 使用Vosk或Whisper进行语音识别
        # 2. 在识别文本中搜索唤醒关键词
        
        # 当前实现：检测到语音能量后，假设可能包含唤醒词
        # 实际项目中应该集成真正的ASR
        energy = self._calculate_energy(audio_data)
        
        if energy > self.energy_threshold * 2:  # 较高的能量阈值
            # 返回第一个唤醒词作为占位符
            # 实际应该返回ASR识别结果
            return self.wake_keywords[0] if self.wake_keywords else None
        
        return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数"""
        if status:
            logger.warning(f"音频流状态: {status}")
        
        # 转换为numpy数组
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 检测语音
        if self._is_speech(audio_data):
            self.audio_buffer.append(audio_data)
            
            # 限制缓冲区大小
            total_duration = len(self.audio_buffer) * self.chunk_size / self.sample_rate
            if total_duration > self.max_buffer_duration:
                self.audio_buffer.pop(0)
            
            # 尝试匹配唤醒词
            # 使用最近1秒的音频
            if len(self.audio_buffer) >= int(self.sample_rate / self.chunk_size):
                recent_audio = np.concatenate(self.audio_buffer[-int(self.sample_rate / self.chunk_size):])
                matched_keyword = self._simple_keyword_match(recent_audio)
                
                if matched_keyword and self.on_wake_detected:
                    logger.info(f"检测到唤醒词: {matched_keyword}")
                    self.on_wake_detected(matched_keyword)
                    self.audio_buffer.clear()  # 清空缓冲区
        else:
            # 静音时清空缓冲区
            if len(self.audio_buffer) > 0:
                self.audio_buffer.clear()
        
        return (None, pyaudio.paContinue)
    
    def start_listening(self, device_index: int = None):
        """
        开始监听唤醒词
        
        Args:
            device_index: 音频设备索引，None表示使用默认设备
        """
        if self.is_listening:
            logger.warning("已经在监听中")
            return
        
        try:
            # 打开音频流
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                input_device_index=device_index
            )
            
            self.is_listening = True
            self._stop_flag.clear()
            self.stream.start_stream()
            
            logger.info("开始监听唤醒词...")
            
        except Exception as e:
            logger.error(f"启动监听失败: {e}")
            raise
    
    def stop_listening(self):
        """停止监听"""
        if not self.is_listening:
            return
        
        self._stop_flag.set()
        self.is_listening = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.warning(f"关闭音频流时出错: {e}")
        
        self.stream = None
        self.audio_buffer.clear()
        logger.info("已停止监听唤醒词")
    
    def cleanup(self):
        """清理资源"""
        self.stop_listening()
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                logger.warning(f"清理PyAudio时出错: {e}")
    
    def list_audio_devices(self):
        """列出可用的音频输入设备"""
        devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        return devices

