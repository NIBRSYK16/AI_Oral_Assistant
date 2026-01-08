# 语音识别模块 (ASR)

基于Vosk的离线语音识别模块，适合树莓派4环境。

## 功能特性

- **离线识别**：完全离线，无需网络连接
- **轻量级**：使用vosk-model-small-en-us-0.15（约40MB），适合树莓派4
- **实时识别**：支持实时音频流识别
- **词级时间戳**：可选返回词级时间戳信息
- **高准确率**：针对英语语音优化

## 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt
```

## 下载Vosk模型

Vosk需要下载预训练模型。推荐使用轻量级模型（适合树莓派4）：

```bash
# 创建模型目录
mkdir -p asr/models

# 下载模型（约40MB）
cd asr/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model-small-en-us-0.15
cd ../..
```

或者访问 [Vosk模型页面](https://alphacephei.com/vosk/models) 手动下载。

**注意**：树莓派4推荐使用 `vosk-model-small-en-us-0.15`，如果需要更高精度可以使用 `vosk-model-en-us-0.22`（约1.8GB，但识别速度会较慢）。

## 使用方法

### 基本使用

```python
from asr.asr_module import SpeechRecognizer

# 初始化识别器
recognizer = SpeechRecognizer()

# 识别音频文件
text = recognizer.recognize_text("audio.wav")
print(f"识别结果: {text}")

# 获取文本和置信度
text, confidence = recognizer.recognize_with_confidence("audio.wav")
print(f"文本: {text}, 置信度: {confidence}")
```

### 获取词级时间戳

```python
result = recognizer.recognize_file("audio.wav", return_words=True)
print(f"文本: {result['text']}")
for word in result['words']:
    print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

### 自定义模型路径

```python
recognizer = SpeechRecognizer(model_path="/path/to/vosk-model")
```

## 配置说明

在 `config.py` 中可以配置：

- **VOSK_MODEL_PATH**: Vosk模型路径
- **SAMPLE_RATE**: 音频采样率（必须与模型匹配，通常为16000）
- **MAX_ALTERNATIVES**: 最大候选结果数
- **WORDS**: 是否返回词级时间戳

## 音频格式要求

- **格式**: WAV
- **采样率**: 16000 Hz（推荐）
- **声道**: 单声道（Mono）
- **位深**: 16位

**注意**：音频必须与模型匹配。如果音频采样率不是16000Hz，需要先进行重采样。

## 与主程序集成

在主程序中使用：

```python
from asr.asr_module import SpeechRecognizer

# 初始化
recognizer = SpeechRecognizer()

# 识别增强后的音频
audio_path = "raspberry_output.wav"  # 来自语音增强模块
text = recognizer.recognize_text(audio_path)
```

## 性能优化

对于树莓派4，建议：

1. **使用轻量级模型**：`vosk-model-small-en-us-0.15`（约40MB）
2. **限制音频长度**：避免处理过长的音频（建议<60秒）
3. **预处理音频**：使用语音增强模块先处理音频，提高识别准确率

## 常见问题

### Q1: 模型加载失败

确保模型路径正确，并且模型文件完整。检查 `asr/models/vosk-model-small-en-us-0.15/` 目录是否存在。

### Q2: 识别结果为空

- 检查音频格式是否正确（16kHz单声道WAV）
- 检查音频是否包含有效语音
- 尝试使用更高质量的模型

### Q3: 识别速度慢

- 使用轻量级模型（vosk-model-small-en-us-0.15）
- 减少音频长度
- 关闭不必要的后台进程

### Q4: 识别准确率低

- 确保音频质量良好（使用语音增强模块）
- 减少背景噪声
- 使用更高质量的模型（但会降低速度）

## 扩展开发

### 支持其他语言

下载对应语言的Vosk模型，例如：
- 中文：vosk-model-cn-0.22
- 其他语言：访问 [Vosk模型页面](https://alphacephei.com/vosk/models)

### 实时识别

可以扩展支持实时音频流识别：

```python
# 实时识别示例（需要扩展）
recognizer.start_stream()
while True:
    text = recognizer.get_partial_result()
    if text:
        print(text)
```

## 参考资料

- Vosk官方文档: https://alphacephei.com/vosk/
- Vosk GitHub: https://github.com/alphacep/vosk-api
- Vosk模型下载: https://alphacephei.com/vosk/models
