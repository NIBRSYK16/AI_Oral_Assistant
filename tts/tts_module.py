"""
TTS语音合成核心模块
基于Piper TTS引擎，适配树莓派环境
"""
import os
import json
import time
import logging
import threading
import queue
from typing import Optional, Generator, Callable, List
import numpy as np

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .config import MODEL_CONFIG, AUDIO_CONFIG, TTS_CONFIG, PERFORMANCE_CONFIG
from .text_processor import TextProcessor
from .audio_player import AudioPlayer

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    TTS语音合成类
    提供文本转语音功能，支持流式合成和播放
    """
    
    def __init__(self, model_name: str = None, auto_load: bool = True):
        """
        初始化TTS模块
        
        Args:
            model_name: 模型名称，默认使用配置中的默认模型
            auto_load: 是否自动加载模型
        """
        self.model_dir = MODEL_CONFIG["model_dir"]
        self.model_name = model_name or MODEL_CONFIG["default_model"]
        self.config_name = self._get_config_name(self.model_name)
        
        # 组件
        self.text_processor = TextProcessor()
        self.audio_player = AudioPlayer()
        
        # 模型相关
        self._voice: Optional[PiperVoice] = None
        self._ort_session: Optional[ort.InferenceSession] = None
        self._model_config: dict = {}
        self._is_loaded = False
        
        # 状态
        self._is_speaking = False
        self._stop_flag = threading.Event()
        
        # 流式合成队列
        self._synthesis_queue = queue.Queue()
        self._synthesis_thread: Optional[threading.Thread] = None
        
        # 回调
        self._on_start: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        self._on_sentence: Optional[Callable[[str], None]] = None
        
        # 配置
        self.speed = TTS_CONFIG["speed"]
        self.sentence_pause = TTS_CONFIG["sentence_pause"]
        
        if auto_load:
            self.load_model()
    
    def _get_config_name(self, model_name: str) -> str:
        """获取模型对应的配置文件名"""
        return model_name + ".json"
    
    def load_model(self, model_name: str = None) -> bool:
        """
        加载TTS模型
        
        Args:
            model_name: 模型名称，为空则使用初始化时指定的模型
            
        Returns:
            是否加载成功
        """
        if model_name:
            self.model_name = model_name
            self.config_name = self._get_config_name(model_name)
        
        model_path = os.path.join(self.model_dir, self.model_name)
        config_path = os.path.join(self.model_dir, self.config_name)
        
        # 检查模型文件
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            logger.info("请下载Piper模型到models目录")
            logger.info("下载地址: https://github.com/rhasspy/piper/blob/master/VOICES.md")
            return False
        
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        try:
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self._model_config = json.load(f)
            
            logger.info(f"正在加载模型: {self.model_name}")
            start_time = time.time()
            
            if PIPER_AVAILABLE:
                # 使用Piper库加载
                self._voice = PiperVoice.load(model_path, config_path)
                logger.info("使用Piper引擎")
            elif ONNX_AVAILABLE:
                # 直接使用ONNX Runtime
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = PERFORMANCE_CONFIG["num_threads"]
                self._ort_session = ort.InferenceSession(model_path, sess_options)
                logger.info("使用ONNX Runtime引擎")
            else:
                logger.error("未安装Piper或ONNX Runtime，无法加载模型")
                return False
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        将文本合成为音频
        
        Args:
            text: 要合成的文本
            
        Returns:
            音频数据(numpy数组)，失败返回None
        """
        if not self._is_loaded:
            logger.error("模型未加载")
            return None
        
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return None
        
        # 文本预处理
        text = self.text_processor.normalize_text(text)
        
        try:
            start_time = time.time()
            
            if self._voice:
                # 使用Piper合成
                audio_data = self._synthesize_piper(text)
            else:
                # 使用ONNX直接合成
                audio_data = self._synthesize_onnx(text)
            
            if audio_data is not None:
                duration = len(audio_data) / AUDIO_CONFIG["sample_rate"]
                synth_time = time.time() - start_time
                rtf = synth_time / duration if duration > 0 else 0
                logger.debug(f"合成完成: {duration:.2f}秒音频, RTF={rtf:.3f}")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"合成失败: {e}")
            return None
    
    def _synthesize_piper(self, text: str) -> np.ndarray:
        """使用Piper引擎合成"""
        import wave
        import io
        
        # Piper输出WAV数据
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            # 使用synthesize_wav方法，它会自动设置WAV格式参数
            self._voice.synthesize_wav(text, wav_file)
        
        # 读取音频数据
        wav_buffer.seek(0)
        with wave.open(wav_buffer, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            # 转换为float32格式（归一化到-1到1）
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_data
    
    def _synthesize_onnx(self, text: str) -> np.ndarray:
        """使用ONNX Runtime直接合成"""
        # 这里需要根据具体的VITS模型实现
        # 简化处理：如果没有Piper，返回静音
        logger.warning("ONNX直接合成暂未完全实现，请安装piper-tts")
        duration = len(text) * 0.1  # 估算时长
        sample_rate = AUDIO_CONFIG["sample_rate"]
        return np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    def speak(self, text: str, blocking: bool = True):
        """
        合成并播放文本
        
        Args:
            text: 要播放的文本
            blocking: 是否阻塞等待播放完成
        """
        if not text or not text.strip():
            return
        
        self._stop_flag.clear()
        self._is_speaking = True
        
        if self._on_start:
            self._on_start()
        
        def _do_speak():
            try:
                # 分句处理
                sentences = self.text_processor.split_sentences(text)
                
                for i, sentence in enumerate(sentences):
                    if self._stop_flag.is_set():
                        break
                    
                    if self._on_sentence:
                        self._on_sentence(sentence)
                    
                    logger.info(f"正在合成: {sentence[:50]}...")
                    
                    # 合成
                    audio = self.synthesize(sentence)
                    
                    if audio is not None and not self._stop_flag.is_set():
                        # 播放
                        self.audio_player.play(audio, blocking=True)
                        
                        # 句子间停顿
                        if i < len(sentences) - 1 and self.sentence_pause > 0:
                            time.sleep(self.sentence_pause)
            
            finally:
                self._is_speaking = False
                if self._on_complete and not self._stop_flag.is_set():
                    self._on_complete()
        
        if blocking:
            _do_speak()
        else:
            thread = threading.Thread(target=_do_speak)
            thread.daemon = True
            thread.start()
    
    def speak_stream(self, text_generator: Generator[str, None, None], blocking: bool = True):
        """
        流式合成和播放
        
        Args:
            text_generator: 文本生成器（如LLM的流式输出）
            blocking: 是否阻塞
        """
        self._stop_flag.clear()
        self._is_speaking = True
        
        if self._on_start:
            self._on_start()
        
        def _do_stream():
            try:
                buffer = ""
                delimiters = TTS_CONFIG["sentence_delimiters"]
                
                for text_chunk in text_generator:
                    if self._stop_flag.is_set():
                        break
                    
                    buffer += text_chunk
                    
                    # 检查是否有完整句子
                    while any(d in buffer for d in delimiters):
                        # 找到第一个分隔符位置
                        split_pos = -1
                        for d in delimiters:
                            pos = buffer.find(d)
                            if pos != -1:
                                if split_pos == -1 or pos < split_pos:
                                    split_pos = pos
                        
                        if split_pos != -1:
                            sentence = buffer[:split_pos + 1].strip()
                            buffer = buffer[split_pos + 1:].strip()
                            
                            if sentence and not self._stop_flag.is_set():
                                if self._on_sentence:
                                    self._on_sentence(sentence)
                                
                                audio = self.synthesize(sentence)
                                if audio is not None:
                                    self.audio_player.play(audio, blocking=True)
                                    if self.sentence_pause > 0:
                                        time.sleep(self.sentence_pause)
                        else:
                            break
                
                # 处理剩余文本
                if buffer.strip() and not self._stop_flag.is_set():
                    audio = self.synthesize(buffer.strip())
                    if audio is not None:
                        self.audio_player.play(audio, blocking=True)
            
            finally:
                self._is_speaking = False
                if self._on_complete and not self._stop_flag.is_set():
                    self._on_complete()
        
        if blocking:
            _do_stream()
        else:
            thread = threading.Thread(target=_do_stream)
            thread.daemon = True
            thread.start()
    
    def stop(self):
        """停止合成和播放"""
        self._stop_flag.set()
        self.audio_player.stop()
        self._is_speaking = False
        logger.info("TTS已停止")
    
    @property
    def is_speaking(self) -> bool:
        """返回当前是否正在播放"""
        return self._is_speaking
    
    @property
    def is_loaded(self) -> bool:
        """返回模型是否已加载"""
        return self._is_loaded
    
    def set_speed(self, speed: float):
        """设置语速 (0.5-2.0)"""
        self.speed = max(0.5, min(2.0, speed))
    
    def set_callbacks(self, on_start: Callable = None, 
                     on_complete: Callable = None,
                     on_sentence: Callable[[str], None] = None):
        """
        设置回调函数
        
        Args:
            on_start: 开始播放时调用
            on_complete: 播放完成时调用
            on_sentence: 每个句子开始播放时调用
        """
        self._on_start = on_start
        self._on_complete = on_complete
        self._on_sentence = on_sentence
    
    def get_available_models(self) -> List[dict]:
        """获取可用的模型列表"""
        available = []
        for key, info in MODEL_CONFIG["available_models"].items():
            model_path = os.path.join(self.model_dir, info["model"])
            exists = os.path.exists(model_path)
            available.append({
                "key": key,
                "model": info["model"],
                "description": info["description"],
                "installed": exists
            })
        return available
    
    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "config": self._model_config,
            "sample_rate": AUDIO_CONFIG["sample_rate"]
        }


def test_tts():
    """测试TTS模块"""
    logging.basicConfig(level=logging.INFO)
    
    print("=== TTS模块测试 ===\n")
    
    tts = TextToSpeech(auto_load=False)
    
    # 检查可用模型
    print("可用模型列表:")
    for model in tts.get_available_models():
        status = "已安装" if model["installed"] else "未安装"
        print(f"  [{status}] {model['key']}: {model['description']}")
    
    print("\n尝试加载模型...")
    if tts.load_model():
        print("模型加载成功!")
        print(f"模型信息: {tts.get_model_info()}")
        
        # 测试合成
        test_text = "Hello, how are you today? I am your English learning assistant."
        print(f"\n测试文本: {test_text}")
        print("正在合成并播放...")
        
        tts.set_callbacks(
            on_start=lambda: print("[开始播放]"),
            on_complete=lambda: print("[播放完成]"),
            on_sentence=lambda s: print(f"[播放句子] {s}")
        )
        
        tts.speak(test_text, blocking=True)
    else:
        print("模型加载失败，请检查模型文件是否存在")
        print(f"模型目录: {tts.model_dir}")


if __name__ == "__main__":
    test_tts()
