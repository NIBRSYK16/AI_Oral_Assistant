#!/usr/bin/env python3
"""
音频录音测试程序
用于测试麦克风录音功能，检测语音活动并播放录制的音频
"""
import numpy as np
import pyaudio
import wave
import time
import sys
import os

# 音频参数
SAMPLE_RATE = 44100  # 使用44100 Hz（适配USB音频设备）
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

# VAD参数
ENERGY_THRESHOLD = 0.01  # 能量阈值
SILENCE_DURATION = 1.0   # 静音持续时间（秒）后停止录音
MIN_RECORDING_DURATION = 0.5  # 最小录音时长（秒）

class AudioRecorder:
    """音频录音器"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = CHANNELS
        self.format = FORMAT
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_data = []
        
    def _calculate_energy(self, audio_chunk):
        """计算音频能量"""
        if len(audio_chunk) == 0:
            return 0.0
        # 转换为float并归一化
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        return np.mean(audio_float ** 2)
    
    def _is_speech(self, audio_chunk):
        """判断是否为语音"""
        energy = self._calculate_energy(audio_chunk)
        return energy > ENERGY_THRESHOLD
    
    def _get_supported_sample_rate(self, device_index=None):
        """获取设备支持的采样率"""
        if device_index is None:
            device_index = self.p.get_default_input_device_info()['index']
        
        device_info = self.p.get_device_info_by_index(device_index)
        default_rate = int(device_info['defaultSampleRate'])
        
        # 常用采样率列表
        preferred_rates = [self.sample_rate, 44100, 48000, 22050, 16000, 32000, 8000]
        
        for rate in preferred_rates:
            try:
                test_stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=device_index
                )
                test_stream.close()
                print(f"✓ 找到支持的采样率: {rate} Hz (设备默认: {default_rate} Hz)")
                return rate
            except Exception as e:
                continue
        
        print(f"⚠ 无法使用目标采样率，使用设备默认: {default_rate} Hz")
        return default_rate
    
    def list_devices(self):
        """列出所有音频输入设备"""
        print("\n" + "="*60)
        print("可用的音频输入设备:")
        print("="*60)
        
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
                print(f"  [{i}] {info['name']}")
                print(f"      通道数: {info['maxInputChannels']}, "
                      f"默认采样率: {int(info['defaultSampleRate'])} Hz")
        
        print("="*60)
        return devices
    
    def record_until_silence(self, device_index=None, max_duration=30):
        """
        录音直到检测到静音
        
        Args:
            device_index: 音频设备索引，None表示使用默认设备
            max_duration: 最大录音时长（秒）
        
        Returns:
            录制的音频数据（numpy数组）
        """
        if device_index is None:
            device_index = self.p.get_default_input_device_info()['index']
        
        # 自动检测采样率
        actual_rate = self._get_supported_sample_rate(device_index)
        if actual_rate != self.sample_rate:
            print(f"⚠ 采样率从 {self.sample_rate} Hz 调整为 {actual_rate} Hz")
            self.sample_rate = actual_rate
        
        print(f"\n开始录音...")
        print(f"  采样率: {self.sample_rate} Hz")
        print(f"  等待语音输入...")
        print(f"  (说话后，静音 {SILENCE_DURATION} 秒后自动停止)")
        print(f"  最大录音时长: {max_duration} 秒")
        print(f"  按 Ctrl+C 可随时中断\n")
        
        self.audio_data = []
        silence_start_time = None
        speech_detected = False
        start_time = time.time()
        
        try:
            # 打开音频流
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index
            )
            
            print("✓ 音频流已打开，开始监听...\n")
            
            while True:
                # 检查最大时长
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    print(f"\n达到最大录音时长 ({max_duration} 秒)，停止录音")
                    break
                
                # 读取音频数据
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"\n⚠ 读取音频数据时出错: {e}")
                    continue
                
                # 转换为numpy数组
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_data.append(audio_chunk)
                
                # 检测语音
                is_speech = self._is_speech(audio_chunk)
                energy = self._calculate_energy(audio_chunk)
                
                # 显示实时状态
                status = "🔊 检测到语音" if is_speech else "🔇 静音"
                print(f"\r{status} | 能量: {energy:.6f} | 时长: {elapsed:.1f}s", end='', flush=True)
                
                if is_speech:
                    speech_detected = True
                    silence_start_time = None
                else:
                    # 如果已经检测到语音，开始计时静音
                    if speech_detected:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        else:
                            silence_duration = time.time() - silence_start_time
                            if silence_duration >= SILENCE_DURATION:
                                print(f"\n\n检测到静音 {SILENCE_DURATION} 秒，停止录音")
                                break
                
        except KeyboardInterrupt:
            print("\n\n用户中断录音")
        except Exception as e:
            print(f"\n\n录音出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
        
        if not self.audio_data:
            print("\n⚠ 没有录制到任何音频数据")
            return None
        
        # 合并所有音频块
        audio_array = np.concatenate(self.audio_data)
        duration = len(audio_array) / self.sample_rate
        
        if duration < MIN_RECORDING_DURATION:
            print(f"\n⚠ 录音时长过短 ({duration:.2f} 秒)，可能没有检测到语音")
            return None
        
        print(f"\n✓ 录音完成！")
        print(f"  总时长: {duration:.2f} 秒")
        print(f"  采样点数: {len(audio_array)}")
        
        return audio_array
    
    def save_wav(self, audio_data, filename="test_recording.wav"):
        """保存音频为WAV文件"""
        if audio_data is None:
            print("⚠ 没有音频数据可保存")
            return False
        
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            print(f"✓ 音频已保存到: {filename}")
            return True
        except Exception as e:
            print(f"✗ 保存音频失败: {e}")
            return False
    
    def play_audio(self, audio_data):
        """播放音频"""
        if audio_data is None:
            print("⚠ 没有音频数据可播放")
            return False
        
        print("\n" + "="*60)
        print("播放录制的音频...")
        print("="*60)
        
        try:
            # 打开输出流
            output_stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # 分块播放
            chunk_size = self.chunk_size
            total_chunks = len(audio_data) // chunk_size + 1
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                # 确保chunk长度是chunk_size
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                output_stream.write(chunk.tobytes())
                progress = min(100, (i + chunk_size) * 100 // len(audio_data))
                print(f"\r播放进度: {progress}%", end='', flush=True)
            
            output_stream.stop_stream()
            output_stream.close()
            
            print("\n✓ 播放完成！")
            return True
            
        except Exception as e:
            print(f"\n✗ 播放音频失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if self.p:
            try:
                self.p.terminate()
            except:
                pass


def main():
    """主函数"""
    print("="*60)
    print("音频录音测试程序")
    print("="*60)
    
    recorder = AudioRecorder()
    
    try:
        # 1. 列出设备
        devices = recorder.list_devices()
        
        # 2. 选择设备（可选）
        print("\n选择音频设备:")
        print("  直接按Enter使用默认设备")
        print("  或输入设备编号")
        
        device_input = input("设备编号: ").strip()
        device_index = None
        if device_input:
            try:
                device_index = int(device_input)
                if device_index < 0 or device_index >= len(devices):
                    print(f"⚠ 无效的设备编号，使用默认设备")
                    device_index = None
            except ValueError:
                print("⚠ 无效输入，使用默认设备")
                device_index = None
        
        if device_index is None:
            device_index = recorder.p.get_default_input_device_info()['index']
            device_name = recorder.p.get_device_info_by_index(device_index)['name']
            print(f"\n使用默认设备: [{device_index}] {device_name}")
        
        # 3. 录音
        audio_data = recorder.record_until_silence(device_index=device_index)
        
        if audio_data is None:
            print("\n⚠ 录音失败，程序退出")
            return
        
        # 4. 保存音频
        filename = "test_recording.wav"
        recorder.save_wav(audio_data, filename)
        
        # 5. 询问是否播放
        print("\n" + "="*60)
        play = input("是否播放录制的音频？(y/n): ").strip().lower()
        
        if play == 'y' or play == 'yes':
            recorder.play_audio(audio_data)
        
        print("\n" + "="*60)
        print("测试完成！")
        print(f"音频文件保存在: {os.path.abspath(filename)}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()

