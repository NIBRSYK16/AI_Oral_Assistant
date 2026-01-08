"""
音频去噪模块
实现谱减法和神经网络去噪算法
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# torch 是可选的（树莓派上可能不安装）
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Denoiser:
    def __init__(self, use_pretrained=True, device='cpu'):
        """
        初始化去噪模型
        
        参数:
            use_pretrained: 是否使用预训练模型
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model = None
        
        if use_pretrained:
            self._load_pretrained_model()
        else:
            self._load_lightweight_model()
    
    def _load_pretrained_model(self):
        """加载预训练模型（在树莓派上可能较慢）"""
        if not TORCH_AVAILABLE:
            # 如果没有torch，直接使用轻量级方法
            self._load_lightweight_model()
            return
        
        try:
            # 尝试加载轻量级模型
            from asteroid.models import BaseModel
            
            # 使用更轻量的模型
            model_name = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
            self.model = BaseModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("加载预训练模型成功")
        except ImportError:
            print("未安装asteroid，使用传统方法")
            self._load_lightweight_model()
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            self._load_lightweight_model()
    
    def _load_lightweight_model(self):
        """加载轻量级模型（适合树莓派）"""
        # 使用传统的谱减法或维纳滤波
        print("使用轻量级去噪方法")
        self.model = "spectral_subtraction"
    
    def denoise(self, audio, sr=16000):
        """
        去噪处理
        
        参数:
            audio: 单通道音频
            sr: 采样率
            
        返回:
            去噪后的音频
        """
        if isinstance(self.model, str) and self.model == "spectral_subtraction":
            # 使用谱减法（计算量小，适合树莓派）
            return self._spectral_subtraction(audio, sr)
        else:
            # 使用神经网络模型
            return self._neural_denoise(audio, sr)
    
    def _spectral_subtraction(self, audio, sr=16000, noise_reduction=0.5):
        """
        经典谱减法（使用librosa，不依赖torch）
        """
        try:
            import librosa
            import scipy.signal as signal
        except ImportError:
            # 如果连librosa都没有，直接返回原音频
            return audio
        
        # 确保是numpy数组
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        
        # 使用librosa计算STFT
        n_fft = 512
        hop_length = 160
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # 计算幅度谱和相位谱
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 估计噪声谱（使用前几帧）
        noise_frames = 10
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # 谱减法
        enhanced_magnitude = magnitude - noise_reduction * noise_estimate
        enhanced_magnitude = np.maximum(enhanced_magnitude, 1e-6)
        
        # 重建复STFT
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        
        # ISTFT
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
        
        return enhanced_audio
    
    def _neural_denoise(self, audio, sr=16000):
        """使用神经网络去噪"""
        if not TORCH_AVAILABLE:
            # 如果没有torch，回退到谱减法
            return self._spectral_subtraction(audio, sr)
        
        try:
            # 转换为模型输入格式
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                enhanced = self.model(audio_tensor)
            
            return enhanced.squeeze().cpu().numpy()
        except Exception as e:
            print(f"神经网络去噪失败: {e}")
            # 回退到谱减法
            return self._spectral_subtraction(audio, sr)