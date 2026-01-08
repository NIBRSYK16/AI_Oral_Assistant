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
                 energy_threshold: float = None,
                 silence_duration: float = 0.5):
        """
        初始化唤醒词检测器
        
        Args:
            wake_keywords: 唤醒关键词列表，默认 ["assistant", "voice assistant"]
            sample_rate: 采样率
            chunk_size: 音频块大小
            energy_threshold: 能量阈值，低于此值认为是静音（None则自动调整）
            silence_duration: 静音持续时间（秒），用于判断语音结束
        """
        self.wake_keywords = wake_keywords or ["assistant", "voice assistant", "语音助手"]
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        # 根据采样率自动调整能量阈值（44100Hz需要更低的阈值）
        if energy_threshold is None:
            if sample_rate >= 44100:
                self.energy_threshold = 0.005  # 44100Hz使用更低的阈值
            else:
                self.energy_threshold = 0.01
        else:
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
        
        # 当前实现：基于能量和持续时间检测
        # 改进：检测到持续的高能量语音（可能是唤醒词）
        energy = self._calculate_energy(audio_data)
        
        # 计算能量变化（检测语音的开始和结束）
        chunk_size = len(audio_data) // 10  # 分成10段
        if chunk_size > 0:
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            chunk_energies = [np.mean(chunk ** 2) for chunk in chunks if len(chunk) > 0]
            
            if len(chunk_energies) > 0:
                avg_energy = np.mean(chunk_energies)
                max_energy = np.max(chunk_energies)
                
                # 检测条件：
                # 1. 平均能量超过阈值
                # 2. 峰值能量明显高于平均值（说明有清晰的语音）
                # 3. 能量变化较大（说明不是持续噪声）
                if avg_energy > self.energy_threshold * 1.5 and max_energy > avg_energy * 1.5:
                    # 返回第一个唤醒词作为占位符
                    # 实际应该返回ASR识别结果
                    return self.wake_keywords[0] if self.wake_keywords else None
        
        return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio回调函数
        优化性能：减少不必要的计算和日志
        """
        # 状态码2表示输入溢出，这在持续录音时很常见，可以忽略
        # 只在严重错误时记录（status != 0 and status != 2）
        if status and status != 2:
            logger.warning(f"音频流状态异常: {status}")
        
        # 快速转换为numpy数组（避免不必要的复制）
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)
        
        # 快速检测语音（只计算能量，不做其他处理）
        energy = np.mean(audio_data ** 2)
        is_speech = energy > self.energy_threshold
        
        if is_speech:
            self.audio_buffer.append(audio_data)
            
            # 限制缓冲区大小（优化：只在必要时清理）
            buffer_frames = len(self.audio_buffer)
            if buffer_frames * self.chunk_size / self.sample_rate > self.max_buffer_duration:
                # 只保留最近的数据
                keep_frames = int(self.max_buffer_duration * self.sample_rate / self.chunk_size)
                self.audio_buffer = self.audio_buffer[-keep_frames:]
            
            # 尝试匹配唤醒词（使用最近约1秒的音频）
            frames_per_second = int(self.sample_rate / self.chunk_size)
            if len(self.audio_buffer) >= frames_per_second:
                # 只处理最近1秒的数据
                recent_frames = self.audio_buffer[-frames_per_second:]
                recent_audio = np.concatenate(recent_frames)
                
                matched_keyword = self._simple_keyword_match(recent_audio)
                
                if matched_keyword and self.on_wake_detected:
                    logger.info(f"检测到唤醒词: {matched_keyword} (能量: {energy:.6f})")
                    try:
                        self.on_wake_detected(matched_keyword)
                    except Exception as e:
                        logger.error(f"唤醒回调函数出错: {e}")
                    # 清空缓冲区，避免重复触发
                    self.audio_buffer.clear()
        else:
            # 静音时，如果缓冲区有数据但时间较短，保留（可能是短暂停顿）
            # 如果静音时间较长，清空缓冲区
            if len(self.audio_buffer) > 0:
                buffer_duration = len(self.audio_buffer) * self.chunk_size / self.sample_rate
                if buffer_duration < 0.3:  # 如果缓冲区数据少于0.3秒，可能是误检测，清空
                    self.audio_buffer.clear()
        
        return (None, pyaudio.paContinue)
    
    def _get_supported_sample_rate(self, device_index: int = None) -> int:
        """
        获取设备支持的采样率
        如果目标采样率不支持，尝试其他常用采样率
        
        Args:
            device_index: 音频设备索引
            
        Returns:
            支持的采样率
        """
        # 常用采样率列表（按优先级排序）
        preferred_rates = [16000, 44100, 48000, 22050, 32000, 8000]
        
        if device_index is None:
            device_index = self.p.get_default_input_device_info()['index']
        
        device_info = self.p.get_device_info_by_index(device_index)
        default_rate = int(device_info['defaultSampleRate'])
        
        # 首先尝试目标采样率
        if self.sample_rate in preferred_rates:
            preferred_rates.insert(0, self.sample_rate)
        
        # 测试每个采样率
        for rate in preferred_rates:
            try:
                # 尝试打开一个测试流
                test_stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=device_index
                )
                test_stream.close()
                logger.info(f"找到支持的采样率: {rate} Hz (设备默认: {default_rate} Hz)")
                return rate
            except Exception:
                continue
        
        # 如果都不支持，使用设备默认采样率
        logger.warning(f"无法使用目标采样率 {self.sample_rate} Hz，使用设备默认采样率: {default_rate} Hz")
        return default_rate
    
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
            # 自动检测并适配采样率
            actual_sample_rate = self._get_supported_sample_rate(device_index)
            
            # 如果采样率改变，更新内部采样率并调整阈值
            if actual_sample_rate != self.sample_rate:
                logger.info(f"采样率从 {self.sample_rate} Hz 调整为 {actual_sample_rate} Hz")
                self.sample_rate = actual_sample_rate
                # 根据新采样率调整能量阈值
                if self.sample_rate >= 44100:
                    self.energy_threshold = 0.005
                else:
                    self.energy_threshold = 0.01
                logger.info(f"能量阈值调整为: {self.energy_threshold}")
            
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
            
            logger.info(f"开始监听唤醒词... (采样率: {self.sample_rate} Hz)")
            
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

