"""
语音识别模块 - 树莓派部署
负责录音、音频增强（波束形成+去噪）和音频保存
"""
import numpy as np
import pyaudio
import wave
import time
import threading
import queue
import logging
import os
from .models.beamformer import Beamformer
from .models.denoiser import Denoiser
from .config import Config
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class RaspberryPiAudioProcessor:
    """树莓派音频处理类"""
    
    def __init__(self):
        # 音频参数
        self.channels = Config.CHANNELS
        self.sample_rate = Config.SAMPLE_RATE
        self.chunk_size = Config.CHUNK_SIZE
        self.format = pyaudio.paInt16
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        
        # 初始化模型
        logger.info("初始化语音增强模型...")
        self.beamformer = Beamformer(
            mic_positions=Config.MIC_POSITIONS,
            fs=self.sample_rate,
            direction=Config.LOOK_DIRECTION
        )
        
        self.denoiser = Denoiser(
            use_pretrained=Config.USE_PRETRAINED,
            device='cpu'
        )
        
        # 音频缓冲区 (使用无限队列以防止高采样率下缓冲区溢出)
        self.audio_buffer = queue.Queue(maxsize=0)
        
        # 控制标志
        self.is_recording = False
        self.is_processing = False
        
        # 输出音频路径
        self.output_audio_path = "raspberry_output.wav"
        
        logger.info("树莓派音频处理器初始化完成")
    
    def list_audio_devices(self):
        """
        列出可用音频设备
        
        Returns:
            设备列表
        """
        logger.info("列出可用音频设备...")
        devices = []
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] >= self.channels:
                device_info = {
                    'index': i,
                    'name': dev_info['name'],
                    'channels': dev_info['maxInputChannels'],
                    'sample_rate': dev_info['defaultSampleRate']
                }
                devices.append(device_info)
                logger.info(f"设备 [{i}]: {dev_info['name']}, 通道: {dev_info['maxInputChannels']}, 采样率: {dev_info['defaultSampleRate']}")
        
        return devices
    
    def find_device(self):
        """
        查找合适的音频输入设备
        优先查找支持配置通道数的设备，否则使用默认设备
        
        Returns:
            设备索引，如果未找到返回None（使用默认设备）
        """
        # 首先尝试查找支持配置通道数的设备
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] >= self.channels:
                logger.info(f"找到支持 {self.channels} 通道的设备: {dev_info['name']}")
                return i
        
        # 如果没找到，使用默认设备
        default_device = self.p.get_default_input_device_info()
        logger.info(f"未找到支持 {self.channels} 通道的设备，使用默认设备: {default_device['name']}")
        return default_device['index']
    
    def record_callback(self, in_data, frame_count, time_info, status):
        """录音回调函数"""
        if self.is_recording:
            # 将数据放入队列
            data = np.frombuffer(in_data, dtype=np.int16)
            
            # 根据通道数处理数据
            if self.channels == 1:
                # 单通道：转换为(1, samples)格式，保持一致性
                data = data.reshape(1, -1)
            else:
                # 多通道：重新整形为(channels, samples)
                data = data.reshape(-1, self.channels).T
            
            if not self.audio_buffer.full():
                self.audio_buffer.put(data)
        
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
        if device_index is None:
            device_index = self.p.get_default_input_device_info()['index']
        
        device_info = self.p.get_device_info_by_index(device_index)
        default_rate = int(device_info['defaultSampleRate'])
        
        # 常用采样率列表（按优先级排序）
        preferred_rates = [self.sample_rate, 44100, 48000, 22050, 32000, 16000, 8000]
        
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
    
    def start_recording(self, device_index=None, duration=None):
        """
        开始录音
        
        Args:
            device_index: 音频设备索引，None则自动查找
            duration: 录音时长（秒），None则手动停止
        """
        logger.info("开始录音...")
        
        if device_index is None:
            device_index = self.find_device()
        
        # 检测设备实际支持的通道数
        device_info = self.p.get_device_info_by_index(device_index)
        max_channels = device_info['maxInputChannels']
        actual_channels = min(self.channels, max_channels)
        
        if actual_channels != self.channels:
            logger.warning(f"设备只支持 {max_channels} 通道，使用 {actual_channels} 通道（配置要求 {self.channels} 通道）")
            # 临时修改通道数
            original_channels = self.channels
            self.channels = actual_channels
        
        # 自动检测并适配采样率
        actual_sample_rate = self._get_supported_sample_rate(device_index)
        if actual_sample_rate != self.sample_rate:
            logger.info(f"采样率从 {self.sample_rate} Hz 调整为 {actual_sample_rate} Hz")
            original_sample_rate = self.sample_rate
            self.sample_rate = actual_sample_rate
        
        try:
            # 打开音频流
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index,
                stream_callback=self.record_callback
            )
            
            self.is_recording = True
            
            # 启动处理线程
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.daemon = True
            process_thread.start()
            
            if duration:
                # 定时停止
                time.sleep(duration)
                self.stop_recording(stream)
            else:
                # 等待用户停止
                input("按Enter键停止录音...")
                self.stop_recording(stream)
        except Exception as e:
            # 如果出错，恢复原始通道数和采样率
            if 'original_channels' in locals():
                self.channels = original_channels
            if 'original_sample_rate' in locals():
                self.sample_rate = original_sample_rate
            logger.error(f"录音启动失败: {e}")
            raise
    
    def stop_recording(self, stream):
        """
        停止录音
        
        Args:
            stream: PyAudio流对象
        """
        logger.info("停止录音...")
        self.is_recording = False
        time.sleep(0.5)  # 等待缓冲区处理完成
        stream.stop_stream()
        stream.close()
    
    def process_audio(self):
        """
        处理音频数据（在独立线程中运行）
        先收集原始音频，录音结束后再统一处理，以避免处理速度跟不上导致丢帧
        """
        self.is_processing = True
        
        logger.info("开始收集音频数据...")
        
        # 原始音频缓冲区
        raw_buffer = []
        
        # 阶段1: 快速收集数据
        while self.is_recording or not self.audio_buffer.empty():
            try:
                # 从队列获取数据
                chunk = self.audio_buffer.get(timeout=0.5)
                raw_buffer.append(chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"收集音频时出错: {e}")
                continue
                
        logger.info(f"音频收集完成，共收集 {len(raw_buffer)} 个数据块，开始后期处理...")

        # 阶段2: 统一处理 (波束形成 + 去噪)
        output_buffer = []
        total_chunks = len(raw_buffer)
        
        for i, chunk in enumerate(raw_buffer):
            try:
                # 1. 波束形成（如果只有1通道，跳过波束形成）
                if self.channels == 1:
                    # 单通道：直接使用，不需要波束形成
                    enhanced = chunk[0]  # 提取单通道数据
                else:
                    # 多通道：执行波束形成
                    enhanced = self.beamformer.delay_and_sum(chunk)
                
                # 2. 确保传入去噪的为浮点数（librosa/torch要求浮点输入），并归一化到 [-1, 1]
                if not isinstance(enhanced, np.ndarray):
                    enhanced = np.array(enhanced)

                # 转为 float32
                if np.issubdtype(enhanced.dtype, np.integer):
                    try:
                        max_val = np.iinfo(enhanced.dtype).max
                    except Exception:
                        max_val = 32767
                    enhanced = enhanced.astype(np.float32) / float(max_val)
                else:
                    enhanced = enhanced.astype(np.float32)

                # 3. 去噪
                final = self.denoiser.denoise(enhanced, self.sample_rate)
                
                # 保存结果
                output_buffer.append(final)
                
                # 简单的进度日志
                if (i + 1) % 50 == 0:
                    logger.info(f"正在处理: {i + 1}/{total_chunks}")
                
            except Exception as e:
                logger.error(f"处理音频时出错: {e}")
                continue
        
        # 保存处理后的音频
        if output_buffer:
            final_audio = np.concatenate(output_buffer)
            self.save_audio(final_audio, self.output_audio_path)
            logger.info(f"音频已保存到: {self.output_audio_path}")
        
        self.is_processing = False
        logger.info("音频处理完成")
    
    def save_audio(self, audio, filename):
        """
        保存音频文件
        
        Args:
            audio: 音频数据（numpy数组）
            filename: 输出文件名
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # 转换为16位整数
            audio_int16 = (audio * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 单通道输出
                wf.setsampwidth(2)  # 2字节 = 16位
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"音频已保存: {filename}")
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        try:
            self.p.terminate()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")

def main():
    """主函数 - 用于独立测试"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 50)
    logger.info("树莓派语音增强系统部署")
    logger.info("=" * 50)
    
    # 创建处理器实例
    processor = RaspberryPiAudioProcessor()
    
    try:
        # 显示设备列表
        processor.list_audio_devices()
        
        # 开始录音（默认设备，录音10秒）
        processor.start_recording(duration=10)
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行错误: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()