# 语音增强模块 (Speech Enhancement)

基于树莓派的语音增强模块，负责录音、音频增强（波束形成+去噪）和音频保存。

**注意**：本模块只负责音频增强，不包含语音识别（ASR）功能。语音识别功能请使用 `asr` 模块。

## 功能特性

- 多通道音频录音（支持6通道麦克风阵列）
- 波束形成（Beamforming）：增强目标方向的声音
- 音频去噪：降低背景噪声
- 实时音频处理
- 适配树莓派环境

## 模块结构

```
se/
├── config.py              # 配置文件
├── raspberry_deploy.py    # 主模块（录音和处理）
├── wake_word_detector.py  # 唤醒词检测器
├── models/                # 模型目录
│   ├── beamformer.py     # 波束形成器
│   └── denoiser.py       # 去噪器
├── utils/                 # 工具模块
│   ├── audio_utils.py    # 音频工具函数
│   └── device_utils.py   # 设备检测工具
├── requirements.txt      # Python依赖
└── README.md             # 本文档
```

## 安装依赖

```bash
# 安装系统依赖（树莓派）
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio

# 安装Python依赖
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from raspberry_deploy import RaspberryPiAudioProcessor

# 创建处理器实例
processor = RaspberryPiAudioProcessor()

# 列出可用音频设备
devices = processor.list_audio_devices()

# 开始录音（录音10秒）
processor.start_recording(duration=10)

# 或者手动停止录音
# processor.start_recording()
# ... 等待用户操作 ...
# processor.stop_recording(stream)

# 处理后的音频保存在 raspberry_output.wav
# 清理资源
processor.cleanup()
```

### 命令行测试

```bash
# 本地测试（使用测试音频文件）
python local_test.py

# 树莓派部署测试（实际录音）
python raspberry_deploy.py
```

## 配置说明

在 `config.py` 中可以配置：

- **音频参数**：采样率、通道数、块大小
- **波束形成参数**：麦克风位置、目标方向、算法类型
- **去噪参数**：是否使用预训练模型

### 麦克风阵列配置

默认配置为6通道环形麦克风阵列，位置在 `Config.MIC_POSITIONS` 中定义。如果使用不同的麦克风配置，需要修改此参数。

## 模块说明

### RaspberryPiAudioProcessor

主处理类，提供以下功能：

- `list_audio_devices()`: 列出可用音频设备
- `find_6ch_device()`: 查找6通道麦克风设备
- `start_recording(duration)`: 开始录音
- `stop_recording(stream)`: 停止录音
- `process_audio()`: 处理音频（波束形成+去噪）
- `save_audio(audio, filename)`: 保存音频文件
- `cleanup()`: 清理资源

### Beamformer

波束形成器，支持以下算法：

- DSB (Delay and Sum Beamforming): 延迟求和波束形成
- MVDR (Minimum Variance Distortionless Response): 最小方差无失真响应

### Denoiser

去噪器，支持：

- 谱减法（轻量级，适合树莓派）
- 神经网络去噪（需要预训练模型）

## 工作流程

1. **录音**：从麦克风阵列采集多通道音频
2. **波束形成**：增强目标方向的声音，抑制其他方向的噪声
3. **去噪**：进一步降低背景噪声
4. **保存**：将处理后的音频保存为WAV文件

## 注意事项

1. **ASR功能**：本模块只负责音频增强，不包含语音识别（ASR）功能。语音识别功能请使用项目中的 `asr` 模块（基于Vosk）。
2. **音频设备**：需要6通道麦克风阵列。如果只有单通道或双通道麦克风，可以修改配置使用单通道模式。
3. **性能**：在树莓派上运行时，建议使用轻量级的去噪算法（谱减法）而不是神经网络模型。
4. **输出格式**：处理后的音频为16kHz单声道WAV格式，可直接用于ASR识别。

## 与主程序集成

在主程序中使用：

```python
from se.raspberry_deploy import RaspberryPiAudioProcessor

processor = RaspberryPiAudioProcessor()
processor.start_recording(duration=45)  # 录音45秒

# 等待处理完成
while processor.is_processing:
    time.sleep(0.1)

# 获取处理后的音频路径
audio_path = processor.output_audio_path
```

## 常见问题

### Q1: 找不到6通道麦克风设备

如果只有单通道或双通道麦克风，可以修改 `config.py` 中的 `CHANNELS` 参数。

### Q2: 录音没有声音

检查：

- 麦克风是否正确连接
- 音频设备权限（可能需要sudo）
- 使用 `aplay -l` 检查设备列表

### Q3: 处理速度慢

- 使用谱减法而不是神经网络去噪
- 减少音频块大小
- 关闭不必要的后台进程

## 扩展开发

### 使用ASR功能

语音识别功能已独立为 `asr` 模块，使用方式：

```python
from asr.asr_module import SpeechRecognizer

# 初始化识别器
recognizer = SpeechRecognizer()

# 识别增强后的音频
text = recognizer.recognize_text("raspberry_output.wav")
```

### 修改波束形成算法

在 `models/beamformer.py` 中添加新的算法实现。

## 参考资料

- PyAudio文档: https://people.csail.mit.edu/hubert/pyaudio/
- 波束形成算法: https://en.wikipedia.org/wiki/Beamforming
- 音频去噪: https://en.wikipedia.org/wiki/Noise_reduction
