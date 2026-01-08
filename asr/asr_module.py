"""
语音识别模块
使用Vosk进行离线语音识别，适合树莓派4
"""
import os
import json
import logging
import wave
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入vosk
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("Vosk未安装，语音识别功能将不可用。请运行: pip install vosk")


class SpeechRecognizer:
    """
    语音识别器
    使用Vosk进行离线语音识别
    """
    
    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        """
        初始化语音识别器
        
        Args:
            model_path: Vosk模型路径，None则使用默认路径
            sample_rate: 音频采样率，必须与模型匹配（通常为16000）
        """
        self.sample_rate = sample_rate
        self.model = None
        self.recognizer = None
        
        if not VOSK_AVAILABLE:
            logger.error("Vosk未安装，无法初始化语音识别器")
            raise ImportError("请先安装vosk: pip install vosk")
        
        # 确定模型路径
        if model_path is None:
            from .config import ASRConfig
            model_path = ASRConfig.VOSK_MODEL_PATH
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            logger.error(f"Vosk模型不存在: {model_path}")
            logger.info("请下载Vosk模型:")
            logger.info("1. 访问 https://alphacephei.com/vosk/models")
            logger.info("2. 下载 vosk-model-small-en-us-0.15 (推荐，约40MB)")
            logger.info("3. 解压到: " + model_path)
            raise FileNotFoundError(f"Vosk模型不存在: {model_path}")
        
        # 加载模型
        try:
            logger.info(f"加载Vosk模型: {model_path}")
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)  # 启用词级时间戳
            logger.info("Vosk模型加载成功")
        except Exception as e:
            logger.error(f"加载Vosk模型失败: {e}")
            raise
    
    def recognize_file(self, audio_file: str, return_words: bool = False) -> Dict:
        """
        识别音频文件
        
        Args:
            audio_file: 音频文件路径（WAV格式，16kHz单声道）
            return_words: 是否返回词级时间戳
            
        Returns:
            识别结果字典，包含:
            - text: 识别的文本
            - confidence: 置信度（如果可用）
            - words: 词级时间戳列表（如果return_words=True）
        """
        if not os.path.exists(audio_file):
            logger.error(f"音频文件不存在: {audio_file}")
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
        
        try:
            # 打开WAV文件
            wf = wave.open(audio_file, "rb")
            
            # 检查音频格式
            if wf.getnchannels() != 1:
                logger.warning(f"音频不是单声道，当前通道数: {wf.getnchannels()}")
            
            if wf.getframerate() != self.sample_rate:
                logger.warning(f"音频采样率不匹配，期望: {self.sample_rate}, 实际: {wf.getframerate()}")
            
            # 读取音频数据
            results = []
            while True:
                data = wf.readframes(4000)  # 每次读取4000帧
                if len(data) == 0:
                    break
                
                # 识别
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if 'text' in result and result['text']:
                        results.append(result)
                else:
                    # 部分结果
                    partial = json.loads(self.recognizer.PartialResult())
                    if 'partial' in partial and partial['partial']:
                        logger.debug(f"部分识别: {partial['partial']}")
            
            # 获取最终结果
            final_result = json.loads(self.recognizer.FinalResult())
            
            wf.close()
            
            # 合并所有结果
            full_text = " ".join([r.get('text', '') for r in results if r.get('text')])
            if final_result.get('text'):
                full_text = full_text + " " + final_result.get('text', '')
            full_text = full_text.strip()
            
            # 构建返回结果
            result_dict = {
                'text': full_text,
                'confidence': final_result.get('confidence', None)
            }
            
            # 如果需要词级时间戳
            if return_words:
                words = []
                for r in results:
                    if 'result' in r:
                        words.extend(r['result'])
                if 'result' in final_result:
                    words.extend(final_result['result'])
                result_dict['words'] = words
            
            logger.info(f"识别完成，文本长度: {len(full_text)} 字符")
            return result_dict
            
        except Exception as e:
            logger.error(f"识别音频文件失败: {e}")
            raise
    
    def recognize_text(self, audio_file: str) -> str:
        """
        识别音频文件并返回文本（简化接口）
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            识别的文本字符串
        """
        result = self.recognize_file(audio_file, return_words=False)
        return result.get('text', '')
    
    def recognize_with_confidence(self, audio_file: str) -> tuple:
        """
        识别音频文件并返回文本和置信度
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            (text, confidence) 元组
        """
        result = self.recognize_file(audio_file, return_words=False)
        return result.get('text', ''), result.get('confidence', None)
    
    def is_available(self) -> bool:
        """
        检查语音识别是否可用
        
        Returns:
            True如果可用，False否则
        """
        return VOSK_AVAILABLE and self.model is not None and self.recognizer is not None


def download_model(model_url: str, model_path: str):
    """
    下载Vosk模型（可选功能）
    
    Args:
        model_url: 模型下载URL
        model_path: 模型保存路径
    """
    try:
        import urllib.request
        import zipfile
        
        logger.info(f"开始下载模型: {model_url}")
        
        # 创建模型目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 下载ZIP文件
        zip_path = model_path + ".zip"
        urllib.request.urlretrieve(model_url, zip_path)
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(model_path))
        
        # 删除ZIP文件
        os.remove(zip_path)
        
        logger.info(f"模型下载完成: {model_path}")
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        raise


if __name__ == "__main__":
    """测试脚本"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) < 2:
        print("用法: python asr_module.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        recognizer = SpeechRecognizer()
        result = recognizer.recognize_file(audio_file, return_words=True)
        
        print("\n识别结果:")
        print(f"文本: {result['text']}")
        if 'confidence' in result and result['confidence']:
            print(f"置信度: {result['confidence']:.2f}")
        if 'words' in result:
            print(f"\n词级时间戳 (前10个):")
            for word in result['words'][:10]:
                print(f"  {word.get('word', '')}: {word.get('start', 0):.2f}s - {word.get('end', 0):.2f}s")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        sys.exit(1)
