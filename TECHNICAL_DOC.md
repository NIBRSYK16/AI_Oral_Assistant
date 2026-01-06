# 树莓派英语口语练习助手 - 技术文档

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [运行流程](#3-运行流程)
4. [模块详解](#4-模块详解)
   - [4.1 语音合成模块（TTS）](#41-语音合成模块tts)
   - [4.2 语音识别模块（ASR）](#42-语音识别模块asr)
   - [4.3 评分模块（Scoring）](#43-评分模块scoring)
5. [关键代码解析](#5-关键代码解析)
6. [配置说明](#6-配置说明)

---

## 1. 项目概述

### 1.1 项目简介

树莓派英语口语练习助手是一个基于树莓派4B的离线英语口语训练系统。该系统集成了语音合成（TTS）、语音识别（ASR）和智能评分三个核心模块，能够实现完整的口语练习流程：题目播放、用户录音、自动评分和反馈生成。

### 1.2 技术特点

- **完全离线运行**：所有功能在本地执行，不依赖网络连接
- **适配树莓派**：针对ARM架构优化，使用轻量级算法
- **模块化设计**：三个核心模块独立，便于维护和扩展
- **基于TOEFL标准**：评分系统参考TOEFL SpeechRater评分准则

### 1.3 技术栈

- **Python 3.9**：主要编程语言
- **Piper TTS**：语音合成引擎
- **PyAudio**：音频输入输出
- **Librosa**：音频处理和分析
- **NumPy/SciPy**：数值计算
- **Conda**：环境管理

---

## 2. 系统架构

### 2.1 整体架构

系统采用分层模块化架构，主要包含以下层次：

```
┌─────────────────────────────────────────┐
│         主程序层 (main.py)              │
│      OralAssistant 主控制类             │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼────┐
│  TTS  │ │  ASR  │ │Scoring │
│ 模块  │ │ 模块  │ │  模块  │
└───────┘ └───────┘ └────────┘
```

### 2.2 模块职责

**主程序层（main.py）**
- 协调三个模块的工作流程
- 管理题库和用户交互
- 控制练习流程（出题、准备、录音、评分）

**TTS模块（tts/）**
- 文本预处理和句子分割
- 调用Piper TTS引擎进行语音合成
- 音频流式播放

**ASR模块（asr/）**
- 多通道音频采集
- 音频增强（波束形成、去噪）
- 音频保存（ASR识别功能待集成）

**评分模块（scoring/）**
- 音频特征提取（VAD、重音、静音检测）
- 文本特征分析（词类型、词频、语法）
- 综合评分和反馈生成

### 2.3 数据流

```
用户输入
   │
   ├─→ [TTS模块] → 题目语音播放
   │
   ├─→ [ASR模块] → 录音 → 音频增强 → 保存音频文件
   │
   └─→ [评分模块] ← 音频文件 + 识别文本
                    │
                    └─→ 评分结果 + 反馈
```

---

## 3. 运行流程

### 3.1 初始化流程

系统启动时，主程序 `main.py` 执行以下初始化步骤：

```python
# main.py 第34-50行
def __init__(self):
    """初始化助手"""
    logger.info("初始化口语训练助手...")
    
    # 初始化各模块
    self.tts = TextToSpeech(auto_load=True)
    self.audio_processor = RaspberryPiAudioProcessor()
    self.speech_rater = SpeechRater()
    
    # 题库路径
    self.question_bank_path = os.path.join(os.path.dirname(__file__), "doc", "口语题库.docx")
    self.questions = self._load_questions()
    
    # 录音文件路径
    self.recorded_audio_path = "recorded_response.wav"
    
    logger.info("口语训练助手初始化完成")
```

**初始化顺序**：
1. 创建TTS模块实例，自动加载Piper TTS模型
2. 创建ASR模块实例，初始化音频设备和增强模型
3. 创建评分模块实例，加载评分器组件
4. 从 `question.md` 文件加载题库
5. 设置录音文件保存路径

### 3.2 主运行流程

主程序采用事件驱动的工作流程：

```python
# main.py 第372-424行
def run(self):
    """
    主运行流程
    """
    try:
        # 1. 等待唤醒
        if not self._wake_up():
            return
        
        # 主循环
        while True:
            # 2. 出题
            question = self._select_question()
            self._present_question(question)
            
            # 3. 等待准备
            self._wait_preparation(15)
            
            # 4. 录音
            text, audio_path = self._record_response(45)
            
            if not text or not audio_path:
                print("录音或识别失败，跳过本次练习")
                continue
            
            # 5. 评分
            try:
                result = self._score_response(audio_path, text)
                
                # 6. 播放评价
                self._present_feedback(result)
                
            except Exception as e:
                logger.error(f"评分过程出错: {e}")
                print(f"评分失败: {e}")
            
            # 7. 询问是否继续
            if not self._ask_continue():
                break
        
        # 8. 说再见
        self._say_goodbye()
```

**流程说明**：

1. **唤醒阶段**：等待用户输入 `start` 或唤醒词
2. **出题阶段**：随机选择题目，通过TTS模块播放
3. **准备阶段**：15秒倒计时，给用户准备时间
4. **录音阶段**：45秒自动录音，ASR模块进行音频增强
5. **评分阶段**：评分模块分析音频和文本，生成评分结果
6. **反馈阶段**：通过TTS播放英文评价
7. **循环判断**：询问用户是否继续练习

---

## 4. 模块详解

### 4.1 语音合成模块（TTS）

#### 4.1.1 模块概述

TTS模块负责将文本转换为语音并播放。采用Piper TTS引擎，支持离线运行和流式输出，适配树莓派环境。

#### 4.1.2 核心类：TextToSpeech

`TextToSpeech` 类是TTS模块的核心，位于 `tts/tts_module.py`：

```python
# tts/tts_module.py 第33-79行
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
```

**关键组件说明**：

- **text_processor**：文本处理器，负责句子分割和预处理
- **audio_player**：音频播放器，负责音频输出
- **_voice**：Piper TTS语音对象
- **_ort_session**：ONNX运行时会话（如果使用ONNX模型）
- **_synthesis_queue**：流式合成队列，支持异步处理

#### 4.1.3 模型加载流程

TTS模块支持两种模型加载方式：Piper原生模型和ONNX模型。

```python
# tts/tts_module.py 第81-130行（简化版）
def load_model(self):
    """加载TTS模型"""
    model_path = os.path.join(self.model_dir, self.model_name)
    config_path = os.path.join(self.model_dir, self.config_name)
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    try:
        # 优先使用Piper原生接口
        if PIPER_AVAILABLE:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._model_config = json.load(f)
            
            self._voice = PiperVoice.load(model_path, config_path)
            self._is_loaded = True
            logger.info(f"成功加载Piper模型: {self.model_name}")
            return True
        
        # 备选：使用ONNX运行时
        elif ONNX_AVAILABLE:
            self._ort_session = ort.InferenceSession(model_path)
            self._is_loaded = True
            logger.info(f"成功加载ONNX模型: {self.model_name}")
            return True
        
        else:
            logger.error("Piper和ONNX均不可用")
            return False
            
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return False
```

**加载策略**：
1. 优先使用Piper原生接口（如果 `piper-tts` 包已安装）
2. 备选使用ONNX运行时（如果只有 `onnxruntime`）
3. 如果都不可用，记录错误并返回False

#### 4.1.4 文本转语音流程

`speak` 方法是TTS模块的核心接口：

```python
# tts/tts_module.py 第132-180行（简化版）
def speak(self, text: str, blocking: bool = True):
    """
    合成并播放语音
    
    Args:
        text: 要合成的文本
        blocking: 是否阻塞等待播放完成
    """
    if not self._is_loaded:
        logger.warning("模型未加载，无法合成语音")
        return
    
    # 文本预处理：分割为句子
    sentences = self.text_processor.split_sentences(text)
    
    if blocking:
        # 阻塞模式：逐句合成并播放
        for sentence in sentences:
            if self._stop_flag.is_set():
                break
            
            audio_data = self._synthesize_sentence(sentence)
            if audio_data is not None:
                self.audio_player.play(audio_data, blocking=True)
                time.sleep(self.sentence_pause)
    else:
        # 非阻塞模式：使用队列异步处理
        for sentence in sentences:
            self._synthesis_queue.put(sentence)
        
        if self._synthesis_thread is None or not self._synthesis_thread.is_alive():
            self._synthesis_thread = threading.Thread(target=self._synthesis_worker)
            self._synthesis_thread.start()
```

**处理流程**：
1. **文本预处理**：通过 `TextProcessor.split_sentences()` 将文本分割为句子
2. **句子合成**：对每个句子调用 `_synthesize_sentence()` 生成音频
3. **音频播放**：通过 `AudioPlayer.play()` 播放音频
4. **模式选择**：支持阻塞和非阻塞两种模式

#### 4.1.5 句子合成实现

```python
# tts/tts_module.py 第182-220行（简化版）
def _synthesize_sentence(self, sentence: str) -> Optional[np.ndarray]:
    """
    合成单个句子的语音
    
    Args:
        sentence: 要合成的句子
        
    Returns:
        音频数据（numpy数组）
    """
    if not sentence.strip():
        return None
    
    try:
        if self._voice is not None:
            # 使用Piper原生接口
            audio_generator = self._voice.synthesize(sentence)
            audio_data = np.concatenate(list(audio_generator), dtype=np.float32)
        elif self._ort_session is not None:
            # 使用ONNX运行时（需要实现ONNX推理逻辑）
            audio_data = self._synthesize_with_onnx(sentence)
        else:
            logger.error("没有可用的模型")
            return None
        
        # 应用语速调节
        if self.speed != 1.0:
            audio_data = self._adjust_speed(audio_data, self.speed)
        
        return audio_data
        
    except Exception as e:
        logger.error(f"合成失败: {e}")
        return None
```

**合成步骤**：
1. 检查模型是否加载
2. 调用Piper或ONNX接口生成音频
3. 应用语速调节（如果speed != 1.0）
4. 返回numpy数组格式的音频数据

---

### 4.2 语音识别模块（ASR）

#### 4.2.1 模块概述

ASR模块负责多通道音频采集、音频增强和音频保存。该模块主要实现音频预处理功能，包括波束形成和去噪，为后续的语音识别提供高质量的音频输入。目前ASR识别功能待集成（可集成whisper或vosk等离线模型）。

#### 4.2.2 核心类：RaspberryPiAudioProcessor

`RaspberryPiAudioProcessor` 类是ASR模块的核心，位于 `asr/raspberry_deploy.py`：

```python
# asr/raspberry_deploy.py 第22-58行
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
        
        # 音频缓冲区
        self.audio_buffer = queue.Queue(maxsize=100)
        
        # 控制标志
        self.is_recording = False
        self.is_processing = False
        
        # 输出音频路径
        self.output_audio_path = "raspberry_output.wav"
        
        logger.info("树莓派音频处理器初始化完成")
```

**关键组件说明**：

- **beamformer**：波束形成器，用于增强目标方向的声音
- **denoiser**：去噪器，用于降低背景噪声
- **audio_buffer**：音频数据队列，用于实时处理
- **is_recording/is_processing**：状态标志，控制录音和处理流程

#### 4.2.3 录音流程

录音采用回调函数模式，支持实时处理：

```python
# asr/raspberry_deploy.py 第112-150行
def start_recording(self, device_index=None, duration=None):
    """
    开始录音
    
    Args:
        device_index: 音频设备索引，None则自动查找
        duration: 录音时长（秒），None则手动停止
    """
    logger.info("开始录音...")
    
    if device_index is None:
        device_index = self.find_6ch_device()
    
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
```

**录音回调函数**：

```python
# asr/raspberry_deploy.py 第99-110行
def record_callback(self, in_data, frame_count, time_info, status):
    """录音回调函数"""
    if self.is_recording:
        # 将数据放入队列
        data = np.frombuffer(in_data, dtype=np.int16)
        # 重新整形为多通道
        data = data.reshape(-1, self.channels).T  # 转换为(channels, samples)
        
        if not self.audio_buffer.full():
            self.audio_buffer.put(data)
    
    return (None, pyaudio.paContinue)
```

**流程说明**：
1. PyAudio以回调模式打开音频流
2. 每次收到音频数据，回调函数将数据放入队列
3. 独立的处理线程从队列取出数据并处理
4. 支持定时停止或手动停止

#### 4.2.4 音频增强处理

音频处理在独立线程中执行，包括波束形成和去噪两个步骤：

```python
# asr/raspberry_deploy.py 第165-204行
def process_audio(self):
    """
    处理音频数据（在独立线程中运行）
    执行波束形成和去噪处理
    """
    self.is_processing = True
    
    logger.info("开始实时处理音频...")
    
    # 保存处理结果的缓冲区
    output_buffer = []
    
    while self.is_recording or not self.audio_buffer.empty():
        try:
            # 从队列获取数据
            chunk = self.audio_buffer.get(timeout=0.5)
            
            # 1. 波束形成
            enhanced = self.beamformer.delay_and_sum(chunk)
            
            # 2. 去噪
            final = self.denoiser.denoise(enhanced, self.sample_rate)
            
            # 保存结果
            output_buffer.append(final)
            
        except queue.Empty:
            continue
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
```

**处理流程**：
1. 从队列获取多通道音频块
2. 波束形成：将6通道音频合并为单通道，增强目标方向
3. 去噪：降低背景噪声
4. 保存处理后的音频到缓冲区
5. 录音结束后，将所有音频块拼接并保存为WAV文件

#### 4.2.5 波束形成算法

波束形成用于增强目标方向的声音，本项目实现了延迟求和（DSB）算法：

```python
# asr/models/beamformer.py 第59-97行
def delay_and_sum(self, multi_channel_audio):
    """
    延迟求和波束形成
    
    参数:
        multi_channel_audio: 多通道音频 (channels, samples)
        
    返回:
        增强后的单通道音频
    """
    num_channels, num_samples = multi_channel_audio.shape
    enhanced = np.zeros(num_samples)
    
    for ch in range(num_channels):
        # 计算延迟（四舍五入到最近的整数）
        delay = int(round(self.delays[ch]))
        
        if delay >= 0:
            # 延迟该通道
            delayed_signal = np.concatenate([
                np.zeros(delay),
                multi_channel_audio[ch, :-delay] if delay > 0 else multi_channel_audio[ch]
            ])
            # 确保长度一致
            delayed_signal = delayed_signal[:num_samples]
        else:
            # 提前该通道
            delayed_signal = multi_channel_audio[ch, -delay:]
            delayed_signal = np.concatenate([
                delayed_signal,
                np.zeros(-delay)
            ])
        
        # 求和
        enhanced += delayed_signal
    
    # 平均
    enhanced = enhanced / num_channels
    return enhanced
```

**算法原理**：
1. 根据麦克风位置和目标方向计算各通道的时延
2. 对各通道信号进行时延对齐
3. 对齐后的信号求和并平均
4. 目标方向的声音同相叠加增强，其他方向的声音部分抵消

#### 4.2.6 去噪算法

去噪模块支持两种方法：谱减法（轻量级）和神经网络去噪（可选）：

```python
# asr/models/denoiser.py 第73-110行（简化版）
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
```

**算法原理**：
1. 对音频进行短时傅里叶变换（STFT），得到时频表示
2. 使用前几帧估计噪声谱
3. 从幅度谱中减去估计的噪声谱
4. 保持相位不变，重建音频信号

---

### 4.3 评分模块（Scoring）

#### 4.3.1 模块概述

评分模块基于TOEFL SpeechRater评分准则，对用户的语音回答进行多维度评分。评分分为两部分：发音评分（Delivery）和内容评分（Language Use），最终生成综合分数和中英文评价。

#### 4.3.2 评分流程

评分模块的主入口是 `SpeechRater.score()` 方法：

```python
# scoring/speech_rater.py 第65-120行（简化版）
def score(self, audio_path: str, text: str, 
          asr_confidence: Optional[float] = None,
          task_type: str = "independent") -> ScoreResult:
    """
    评分主接口
    
    Args:
        audio_path: 音频文件路径（必需）
        text: 识别得到的文本（必需）
        asr_confidence: ASR平均置信度（可选）
        task_type: 任务类型（"independent" 或 "integrated"）
        
    Returns:
        ScoreResult对象
    """
    try:
        # 1. 音频分析
        audio = self.audio_analyzer.load_audio(audio_path)
        duration = self.audio_analyzer.get_audio_duration(audio)
        
        # 2. 计算发音特征（12个指标）
        delivery_features = self.delivery_scorer.calculate_all_features(
            audio_path, text, asr_confidence
        )
        
        # 3. 计算内容特征（6个指标）
        language_features = self.language_scorer.calculate_all_features(
            text, duration, task_type
        )
        
        # 4. 计算分数
        delivery_score = self.delivery_scorer.calculate_score(delivery_features)
        language_score = self.language_scorer.calculate_score(language_features)
        
        # 5. 计算最终分数
        final_score = self.score_calculator.calculate_final_score(
            delivery_score, language_score
        )
        
        # 6. 生成评价
        score_result_dict = self.score_calculator.get_score_breakdown(
            delivery_score, language_score, delivery_features, language_features
        )
        feedback = self.feedback_generator.generate_feedback(score_result_dict)
        
        # 7. 返回结果
        return ScoreResult(
            raw_score=final_score,
            delivery_score=delivery_score * 4,  # 转换为0-4分
            language_score=language_score * 4,
            delivery_features=delivery_features,
            language_features=language_features,
            feedback_en=feedback["en"],
            feedback_zh=feedback["zh"]
        )
```

**评分步骤**：
1. 加载音频并分析时长
2. 提取12个发音特征（语速、语音块、静音、重音等）
3. 提取6个内容特征（词类型、词频、语法等）
4. 分别计算发音分数和内容分数（0-1范围）
5. 加权求和得到最终分数（0-4分）
6. 根据分数和特征生成中英文评价

#### 4.3.3 音频分析器

`AudioAnalyzer` 负责从音频中提取各种特征：

```python
# scoring/audio_analyzer.py 第49-96行
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
```

**VAD算法**：
1. 将音频分帧，计算每帧的短时能量
2. 使用自适应阈值（平均能量的10%）判断语音/非语音
3. 合并相邻的语音段，过滤过短的段

#### 4.3.4 发音评分器

`DeliveryScorer` 计算12个发音相关指标：

```python
# scoring/delivery_scorer.py 第22-80行（简化版）
def calculate_all_features(self, audio_path: str, text: str, 
                          asr_confidence: Optional[float] = None) -> Dict[str, float]:
    """
    计算所有发音特征
    
    Returns:
        特征字典，包含12个指标的值
    """
    # 加载音频
    audio = self.audio_analyzer.load_audio(audio_path)
    duration = self.audio_analyzer.get_audio_duration(audio)
    
    # 文本预处理
    words = text.lower().split()
    
    # 计算各个特征
    features = {}
    
    # 1. wpsecutt - 语速（每秒单词数）
    features["wpsecutt"] = self._calculate_speaking_rate(words, duration)
    
    # 2. wdpchk - 平均语音块长度
    # 3. wdpchkmeandev - 语音块长度偏差
    chunk_lengths, chunk_deviation = self.audio_analyzer.calculate_speech_chunks(audio, words)
    features["wdpchk"] = np.mean(chunk_lengths) if chunk_lengths else 0.0
    features["wdpchkmeandev"] = chunk_deviation
    
    # 4. silpwd - 短静音频率（>0.15s）
    # 5. longpfreq - 长静音频率（>0.5s）
    short_pauses, long_pauses = self.audio_analyzer.detect_silences(audio)
    features["silpwd"] = len(short_pauses)
    features["longpfreq"] = len(long_pauses)
    
    # 6. repfreq - 重复频率
    features["repfreq"] = self._calculate_repetition_frequency(words)
    
    # 7. ipc - 中断点频率（自我修正）
    features["ipc"] = self._calculate_interruption_points(text)
    
    # 8. dpsec - 不流畅词频率
    features["dpsec"] = self._calculate_disfluency_frequency(words, duration)
    
    # 9. stretimemean - 重音音节平均距离
    # 10. stresyllmdev - 重音音节距离偏差
    stress_times = self.audio_analyzer.detect_stressed_syllables(audio)
    stress_mean, stress_dev = self.audio_analyzer.calculate_stress_intervals(stress_times)
    features["stretimemean"] = stress_mean
    features["stresyllmdev"] = stress_dev
    
    # 11. conftimeavg - ASR置信度
    features["conftimeavg"] = asr_confidence if asr_confidence is not None else 0.5
    
    # 12. L6 - 发音质量评分
    features["L6"] = self._calculate_pronunciation_quality(features)
    
    return features
```

**12个发音指标说明**：

| 指标 | 说明 | 计算方法 |
|------|------|----------|
| wpsecutt | 语速 | 单词数 / 音频时长 |
| wdpchk | 平均语音块长度 | 连续语音段的平均单词数 |
| wdpchkmeandev | 语音块长度偏差 | 语音块长度的标准差 |
| silpwd | 短静音频率 | 时长>0.15s的静音次数 |
| longpfreq | 长静音频率 | 时长>0.5s的静音次数 |
| repfreq | 重复频率 | 重复单词的比例 |
| ipc | 中断点频率 | 自我修正的次数 |
| dpsec | 不流畅词频率 | "um", "uh"等词的频率 |
| stretimemean | 重音平均距离 | 重音音节之间的平均时间间隔 |
| stresyllmdev | 重音距离偏差 | 重音间隔的标准差 |
| conftimeavg | ASR置信度 | 语音识别的平均置信度 |
| L6 | 发音质量 | 综合多个特征的评分 |

#### 4.3.5 内容评分器

`LanguageScorer` 计算6个内容相关指标：

```python
# scoring/language_scorer.py 第56-100行（简化版）
def calculate_all_features(self, text: str, audio_duration: float,
                          task_type: str = "independent") -> Dict[str, float]:
    """
    计算所有语言使用特征
    
    Returns:
        特征字典，包含6个指标的值
    """
    # 文本预处理
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return {key: 0.0 for key in self.weights.keys()}
    
    features = {}
    
    # 1. types - 词类型数量
    features["types"] = self._calculate_word_types(text)
    
    # 2. tpsec - 每秒词类型数
    features["tpsec"] = features["types"] / audio_duration if audio_duration > 0 else 0.0
    
    # 3. poscvamax - POS bigram对比（简化实现）
    features["poscvamax"] = self._calculate_pos_bigram_score(text)
    
    # 4. logfreq - 词汇频率
    features["logfreq"] = self._calculate_log_frequency(words)
    
    # 5. lmscore - 语言模型分数（语法正确性）
    features["lmscore"] = self._calculate_lm_score(text)
    
    # 6. cvamax - 单词数对比
    target_count = (TEXT_CONFIG["target_word_count_independent"] 
                  if task_type == "independent" 
                  else TEXT_CONFIG["target_word_count_integrated"])
    features["cvamax"] = self._calculate_word_count_score(word_count, target_count)
    
    return features
```

**6个内容指标说明**：

| 指标 | 说明 | 计算方法 |
|------|------|----------|
| types | 词类型数量 | 不同词性（POS）的数量 |
| tpsec | 每秒词类型数 | 词类型数 / 音频时长 |
| poscvamax | POS bigram对比 | 词性搭配的多样性 |
| logfreq | 词汇频率 | 使用词汇的平均对数频率 |
| lmscore | 语言模型分数 | 语法正确性评分 |
| cvamax | 单词数对比 | 实际单词数与目标单词数的比值 |

#### 4.3.6 分数计算

`ScoreCalculator` 负责将特征值转换为分数：

```python
# scoring/score_calculator.py 第22-43行
def calculate_final_score(self, delivery_score: float, language_score: float) -> float:
    """
    计算最终分数
    
    Args:
        delivery_score: 发音分数（0-1）
        language_score: 内容分数（0-1）
        
    Returns:
        最终分数（0-4）
    """
    # 加权求和
    combined_score = (delivery_score * self.delivery_weight + 
                     language_score * self.language_weight)
    
    # 缩放到0-4分
    final_score = combined_score * self.max_score
    
    # 确保在有效范围内
    final_score = max(self.min_score, min(self.max_score, final_score))
    
    return round(final_score, 2)
```

**计算流程**：
1. 发音分数和内容分数分别通过加权求和计算（0-1范围）
2. 两部分按权重（默认各50%）合并
3. 缩放到0-4分制
4. 限制在有效范围内

#### 4.3.7 评价生成

`FeedbackGenerator` 根据评分结果生成中英文评价：

```python
# scoring/feedback_generator.py 第48-80行（简化版）
def _generate_english_feedback(self, final_score: float, delivery_score: float,
                               language_score: float, score_result: Dict) -> str:
    """生成英文评价"""
    feedback_parts = []
    
    # 总体评价
    if final_score >= 3.5:
        feedback_parts.append("Excellent performance! Your speaking demonstrates strong fluency and clear expression.")
    elif final_score >= 3.0:
        feedback_parts.append("Good job! Your response shows solid speaking skills with room for improvement.")
    elif final_score >= 2.5:
        feedback_parts.append("Fair performance. You've made progress, but there are areas to work on.")
    elif final_score >= 2.0:
        feedback_parts.append("Your response needs improvement. Focus on both pronunciation and content.")
    else:
        feedback_parts.append("Keep practicing! Focus on the basics of pronunciation and grammar.")
    
    # 发音评价
    delivery_features = score_result.get("delivery_features", {})
    if delivery_score >= 3.0:
        feedback_parts.append("Your pronunciation is clear and your speaking pace is appropriate.")
    elif delivery_score >= 2.0:
        feedback_parts.append("Your pronunciation is understandable, but try to speak more fluently with fewer pauses.")
    else:
        feedback_parts.append("Work on your pronunciation and try to reduce long pauses and repetitions.")
    
    # 内容评价
    if language_score >= 3.0:
        feedback_parts.append("Your language use is effective with good vocabulary and grammar.")
    elif language_score >= 2.0:
        feedback_parts.append("Your language use is adequate, but try to use more varied vocabulary and complex sentences.")
    else:
        feedback_parts.append("Focus on using more diverse vocabulary and improving your grammar.")
    
    return " ".join(feedback_parts)
```

**评价生成策略**：
1. 根据总分确定总体评价等级
2. 分别评价发音和内容两个方面
3. 提供具体的改进建议
4. 生成中英文两套评价文本

---

## 5. 关键代码解析

### 5.1 主程序控制流程

主程序 `OralAssistant` 类负责协调三个模块，实现完整的练习流程。以下是关键方法的实现：

#### 5.1.1 出题与播放

```python
# main.py 第198-194行
def _present_question(self, question: str):
    """
    出题：合成并播放题目
    """
    logger.info(f"出题: {question}")
    
    # 合成题目文本
    question_text = f"{question} You have 15 seconds to prepare and 45 seconds to answer."
    
    # 播放题目
    print(f"\n题目: {question}")
    self.tts.speak(question_text, blocking=True)
```

**实现要点**：
- 将题目文本与时间提示合并
- 使用TTS模块的 `speak()` 方法，阻塞模式确保完整播放
- 同时在控制台显示题目文本

#### 5.1.2 录音与处理

```python
# main.py 第213-268行
def _record_response(self, duration: int = 45):
    """
    录音并识别
    
    Args:
        duration: 录音时长（秒）
        
    Returns:
        (recognized_text, audio_path) 元组
    """
    logger.info(f"开始录音: {duration}秒")
    print(f"\n开始录音，请开始答题...")
    print(f"录音时长: {duration}秒")
    
    try:
        # 启动录音
        self.audio_processor.start_recording(duration=duration)
        
        # 等待录音完成
        while self.audio_processor.is_recording:
            time.sleep(0.1)
        
        # 等待处理完成
        while self.audio_processor.is_processing:
            time.sleep(0.1)
        
        # 获取处理后的音频路径
        audio_path = "raspberry_output.wav"
        
        if not os.path.exists(audio_path):
            logger.warning("音频文件不存在，使用默认路径")
            audio_path = self.recorded_audio_path
        
        # TODO: 调用ASR识别
        recognized_text = self._recognize_speech(audio_path)
        
        logger.info(f"识别结果: {recognized_text}")
        print(f"\n识别结果: {recognized_text}")
        
        return recognized_text, audio_path
        
    except Exception as e:
        logger.error(f"录音失败: {e}")
        print(f"录音失败: {e}")
        return "", ""
```

**实现要点**：
- 使用轮询方式等待录音和处理完成（`is_recording` 和 `is_processing` 标志）
- 检查音频文件是否存在，提供容错机制
- 目前ASR识别功能待集成，返回模拟文本

#### 5.1.3 评分与反馈

```python
# main.py 第288-330行
def _score_response(self, audio_path: str, text: str) -> ScoreResult:
    """
    评分
    
    Args:
        audio_path: 音频文件路径
        text: 识别文本
        
    Returns:
        评分结果
    """
    logger.info("开始评分...")
    print("\n正在评分，请稍候...")
    
    try:
        result = self.speech_rater.score(
            audio_path=audio_path,
            text=text,
            asr_confidence=None,  # 如果ASR提供置信度，可以传入
            task_type="independent"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"评分失败: {e}")
        print(f"评分失败: {e}")
        raise
```

**实现要点**：
- 调用评分模块的统一接口 `score()`
- 传入音频路径和识别文本
- 支持传入ASR置信度（如果可用）
- 异常处理和日志记录

### 5.2 TTS模块关键实现

#### 5.2.1 文本预处理

`TextProcessor` 类负责文本的预处理，确保TTS引擎能够正确合成：

```python
# tts/text_processor.py 第51-79行
def normalize_text(self, text: str) -> str:
    """
    文本正则化处理
    
    Args:
        text: 原始文本
        
    Returns:
        正则化后的文本
    """
    if not text:
        return ""
    
    # 去除多余空白
    text = " ".join(text.split())
    
    # 展开缩写
    text = self._expand_abbreviations(text)
    
    # 处理数字
    text = self._convert_numbers(text)
    
    # 处理货币符号
    text = self._convert_currency(text)
    
    # 处理特殊字符
    text = self._clean_special_chars(text)
    
    return text.strip()
```

**处理步骤**：
1. 去除多余空白字符
2. 展开常见缩写（如 "Mr." → "Mister"）
3. 将数字转换为英文单词（如 "10" → "ten"）
4. 转换货币符号（如 "$10" → "ten dollars"）
5. 清理特殊字符

#### 5.2.2 句子分割

```python
# tts/text_processor.py 第151-184行
def split_sentences(self, text: str) -> List[str]:
    """
    将文本分割成句子
    
    Args:
        text: 输入文本
        
    Returns:
        句子列表
    """
    if not text:
        return []
    
    # 先进行正则化
    text = self.normalize_text(text)
    
    # 使用正则表达式分句
    pattern = r'(?<=[.!?;])\s+'
    sentences = re.split(pattern, text)
    
    # 处理过长的句子
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > self.max_sentence_length:
            # 按逗号或空格进一步分割
            sub_sentences = self._split_long_sentence(sentence)
            result.extend(sub_sentences)
        else:
            result.append(sentence)
    
    return result
```

**分割策略**：
1. 使用正则表达式按标点符号（. ! ? ;）分割
2. 检查句子长度，超过阈值则进一步分割
3. 优先按逗号分割，其次按空格分割

### 5.3 ASR模块关键实现

#### 5.3.1 延迟计算

波束形成需要计算各麦克风相对于参考点的时延：

```python
# asr/models/beamformer.py 第34-57行
def _calculate_delays(self):
    """计算各麦克风相对于参考点的延迟"""
    # 假设声源在远场
    # 方向向量
    direction_vec = np.array([
        np.cos(self.direction),
        np.sin(self.direction),
        0
    ])
    
    # 计算各麦克风的时延（以采样点为单位）
    delays = []
    for pos in self.mic_positions:
        # 投影到方向向量
        proj = np.dot(pos, direction_vec)
        # 转换为时间延迟（秒）
        time_delay = proj / self.c
        # 转换为采样点延迟
        sample_delay = time_delay * self.fs
        delays.append(sample_delay)
    
    # 调整为相对于第一个麦克风的相对延迟
    delays = np.array(delays) - delays[0]
    return delays
```

**计算原理**：
1. 根据目标方向计算方向向量
2. 将麦克风位置投影到方向向量，得到距离差
3. 根据声速计算时间延迟
4. 转换为采样点延迟（乘以采样率）
5. 调整为相对延迟（以第一个麦克风为参考）

#### 5.3.2 实时音频处理

ASR模块使用队列和线程实现实时处理：

```python
# asr/raspberry_deploy.py 第165-204行
def process_audio(self):
    """
    处理音频数据（在独立线程中运行）
    执行波束形成和去噪处理
    """
    self.is_processing = True
    
    logger.info("开始实时处理音频...")
    
    # 保存处理结果的缓冲区
    output_buffer = []
    
    while self.is_recording or not self.audio_buffer.empty():
        try:
            # 从队列获取数据
            chunk = self.audio_buffer.get(timeout=0.5)
            
            # 1. 波束形成
            enhanced = self.beamformer.delay_and_sum(chunk)
            
            # 2. 去噪
            final = self.denoiser.denoise(enhanced, self.sample_rate)
            
            # 保存结果
            output_buffer.append(final)
            
        except queue.Empty:
            continue
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
```

**设计要点**：
- 使用独立线程处理，不阻塞录音
- 队列机制解耦录音和处理
- 异常处理确保稳定性
- 录音结束后拼接所有音频块

### 5.4 评分模块关键实现

#### 5.4.1 特征归一化

评分模块需要将各种特征值归一化到0-1范围，然后加权求和：

```python
# scoring/delivery_scorer.py 第82-143行（简化版）
def calculate_score(self, features: Dict[str, float]) -> float:
    """
    根据特征计算发音分数（0-1）
    
    Args:
        features: 特征字典
        
    Returns:
        归一化后的分数（0-1）
    """
    score = 0.0
    
    for feature_name, feature_value in features.items():
        if feature_name not in self.weights:
            continue
        
        # 归一化特征值（根据经验阈值）
        normalized_value = self._normalize_feature(feature_name, feature_value)
        
        # 加权求和
        score += normalized_value * self.weights[feature_name]
    
    # 确保在0-1范围内
    return max(0.0, min(1.0, score))
```

**归一化策略**：
- 不同特征有不同的取值范围
- 使用经验阈值进行线性或非线性归一化
- 加权求和后限制在0-1范围

#### 5.4.2 重音检测

重音检测用于评估语音的韵律特征：

```python
# scoring/audio_analyzer.py 第130-182行（简化版）
def detect_stressed_syllables(self, audio: np.ndarray) -> List[float]:
    """
    检测重音音节的时间位置
    
    Args:
        audio: 音频数据
        
    Returns:
        重音时间点列表（秒）
    """
    # 计算短时能量
    frame_length = STRESS_CONFIG["frame_length"]
    hop_length = STRESS_CONFIG["hop_length"]
    
    energy = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy.append(np.mean(frame ** 2))
    
    energy = np.array(energy)
    
    # 计算基频（F0）
    f0 = librosa.yin(audio, 
                     fmin=STRESS_CONFIG["f0_range"][0],
                     fmax=STRESS_CONFIG["f0_range"][1])
    
    # 归一化
    energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
    
    # 检测重音：能量和基频都较高的位置
    threshold = STRESS_CONFIG["energy_threshold"]
    stress_times = []
    
    for i, (e, f) in enumerate(zip(energy_norm, f0)):
        if e > threshold and not np.isnan(f):
            time = i * hop_length / self.sample_rate
            stress_times.append(time)
    
    return stress_times
```

**检测原理**：
1. 计算短时能量，识别能量较高的帧
2. 计算基频（F0），识别音调变化
3. 结合能量和基频信息，识别重音位置
4. 返回重音时间点列表

---

## 6. 配置说明

### 6.1 TTS模块配置

TTS模块的配置文件位于 `tts/config.py`：

```python
# tts/config.py 第10-36行
MODEL_CONFIG = {
    # Piper模型目录
    "model_dir": os.path.join(BASE_DIR, "models"),
    # 默认使用的模型 (美音女声-中等质量，平衡质量与性能)
    "default_model": "en_US-amy-medium.onnx",
    # 对应的配置文件
    "default_config": "en_US-amy-medium.onnx.json",
    # 备选模型列表
    "available_models": {
        "amy_medium": {
            "model": "en_US-amy-medium.onnx",
            "config": "en_US-amy-medium.onnx.json",
            "description": "美音女声-中等质量"
        },
        # ... 其他模型
    }
}

# 音频配置
AUDIO_CONFIG = {
    # 采样率 (Piper默认22050Hz)
    "sample_rate": 22050,
    # 音频格式
    "format": "int16",
    # 声道数 (单声道)
    "channels": 1,
    # 音频缓冲区大小 (帧数)
    "buffer_size": 1024,
    # 音频块大小 (用于流式播放)
    "chunk_size": 4096,
}

# TTS引擎配置
TTS_CONFIG = {
    # 语速调节 (0.5-2.0, 1.0为正常速度)
    "speed": 1.0,
    # 是否启用流式输出
    "streaming": True,
    # 句子分割的标点符号
    "sentence_delimiters": [".", "!", "?", ";"],
    # 最大句子长度 (字符数，超过则强制分割)
    "max_sentence_length": 200,
    # 句子间停顿时间 (秒)
    "sentence_pause": 0.3,
}
```

**关键配置项说明**：
- `default_model`：默认使用的TTS模型，可替换为其他模型
- `speed`：语速调节，1.0为正常速度，可调整为0.5-2.0
- `max_sentence_length`：超过此长度的句子会被强制分割
- `sentence_pause`：句子之间的停顿时间

### 6.2 ASR模块配置

ASR模块的配置文件位于 `asr/config.py`：

```python
# asr/config.py 第12-41行
class Config:
    # 音频参数
    SAMPLE_RATE = 16000
    CHANNELS = 6
    CHUNK_SIZE = 1024
    
    # 波束形成参数
    BEAMFORMER_TYPE = 'MVDR'  # 可选: 'DSB'(延迟求和), 'MVDR', 'GSC'
    LOOK_DIRECTION = 0  # 目标方向（弧度）
    MIC_POSITIONS = np.array([  # 假设麦克风为环形阵列
        [0.02, 0, 0],
        [0.01, 0.0173, 0],
        [-0.01, 0.0173, 0],
        [-0.02, 0, 0],
        [-0.01, -0.0173, 0],
        [0.01, -0.0173, 0]
    ])
    
    # 模型路径
    DENOISER_MODEL = "microsoft/asteroid_dprnntasnet-ks2_enh_v2"
    USE_PRETRAINED = True
    
    # 处理参数
    FRAME_LENGTH = 400  # 25ms
    HOP_LENGTH = 160    # 10ms
```

**关键配置项说明**：
- `CHANNELS`：麦克风通道数，默认6通道
- `BEAMFORMER_TYPE`：波束形成算法类型，可选DSB、MVDR、GSC
- `LOOK_DIRECTION`：目标方向（弧度），0表示正前方
- `MIC_POSITIONS`：麦克风位置数组，需要根据实际硬件调整
- `USE_PRETRAINED`：是否使用预训练去噪模型（树莓派上建议False）

### 6.3 评分模块配置

评分模块的配置文件位于 `scoring/config.py`：

```python
# scoring/config.py 第10-33行
# 评分权重配置（基于TOEFL SpeechRater研究）
DELIVERY_WEIGHTS = {
    "stretimemean": 0.15,      # 重音音节平均距离
    "wpsecutt": 0.15,          # 语速（每秒单词数）
    "wdpchk": 0.13,            # 平均语音块长度
    "wdpchkmeandev": 0.13,     # 语音块长度偏差
    "conftimeavg": 0.12,       # ASR置信度
    "repfreq": 0.08,           # 重复频率
    "silpwd": 0.06,            # 短静音频率（>0.15s）
    "ipc": 0.06,               # 中断点频率
    "stresyllmdev": 0.05,      # 重音音节距离偏差
    "L6": 0.03,                # 发音质量评分
    "longpfreq": 0.03,         # 长静音频率（>0.5s）
    "dpsec": 0.01,             # 不流畅词频率
}

LANGUAGE_WEIGHTS = {
    "types": 0.35,             # 词类型数量
    "poscvamax": 0.18,         # POS bigram对比
    "logfreq": 0.15,           # 词汇频率
    "lmscore": 0.11,           # 语言模型分数
    "tpsec": 0.11,             # 每秒词类型数
    "cvamax": 0.10,            # 单词数对比
}

# 评分阈值
SCORE_CONFIG = {
    "max_score": 4.0,          # 满分
    "min_score": 0.0,          # 最低分
    "delivery_weight": 0.5,    # 发音部分权重
    "language_weight": 0.5,    # 内容部分权重
}
```

**权重配置说明**：
- 权重总和应为1.0（或接近1.0）
- 权重越大，该特征对最终分数的影响越大
- 可以根据实际需求调整权重，例如更重视发音或更重视内容
- `delivery_weight` 和 `language_weight` 控制发音和内容的相对重要性

### 6.4 配置调整建议

**性能优化**：
- 树莓派上建议使用低质量TTS模型（`amy_low`）以提高速度
- ASR模块的 `USE_PRETRAINED` 设为False，使用轻量级谱减法
- 减少评分模块的VAD帧长度，降低计算量

**准确性优化**：
- 使用高质量TTS模型（`amy_medium` 或 `ryan_medium`）
- 调整波束形成的 `LOOK_DIRECTION` 以匹配实际声源方向
- 根据实际数据调整评分权重

**功能扩展**：
- 修改 `MIC_POSITIONS` 以适配不同的麦克风阵列布局
- 添加新的TTS模型到 `available_models`
- 调整 `TEXT_CONFIG` 中的目标单词数以适应不同任务类型

---

**技术文档完成**

本文档涵盖了项目的完整技术细节，包括系统架构、运行流程、三个核心模块的详细实现、关键代码解析和配置说明。所有代码引用均标注了文件路径和行号，便于查阅和验证。

