# 树莓派英语口语练习助手

基于树莓派的英语口语训练系统，集成语音合成、语音识别和智能评分功能。

## 功能特性

- **语音合成（TTS）**：将题目文本转换为语音播放
- **语音识别（ASR）**：录音并识别用户回答，支持音频增强
- **智能评分**：基于TOEFL SpeechRater准则进行评分，生成中英文评价

## 系统要求

- 硬件：树莓派4B（推荐4GB内存以上）
- 系统：Raspberry Pi OS 64-bit
- Python：3.9（通过conda管理）
- 已安装：conda（Miniconda或Anaconda）

## 快速安装

### 一键安装（推荐）

```bash
cd /home/pi/Desktop/AI_Oral_Assistant
./install.sh
```

### 手动安装

如果自动安装失败，可以手动执行：

```bash
# 1. 安装系统依赖
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg gfortran build-essential

# 2. 创建conda环境
conda env create -f environment.yml
conda activate oral_assistant

# 3. 安装Python依赖
cd tts && pip install -r requirements.txt && cd ..
cd asr && pip install numpy scipy soundfile librosa matplotlib pyaudio sounddevice && cd ..
cd scoring && pip install numpy scipy librosa soundfile wordfreq nltk && cd ..

# 4. 下载TTS模型
cd tts && python download_models.py en_US-amy-medium && cd ..
```

## 使用方法

### 启动程序

```bash
conda activate oral_assistant
cd /home/pi/Desktop/AI_Oral_Assistant
python main.py
```

**启用/禁用唤醒词**：
```bash
# 启用唤醒词（默认，持续监听模式）
python main.py

# 禁用唤醒词（使用文本输入模式）
python main.py --no-wake-word
```

### 使用流程

1. **启动程序**：运行 `python main.py`
2. **唤醒系统**：说出"voice assistant"或"语音助手"（如果启用唤醒词），或输入 `start`
3. **听题**：系统会播放题目
4. **准备**：15秒准备时间
5. **答题**：45秒答题时间（系统自动录音）
6. **评分**：系统自动评分并播放评价
7. **继续**：选择是否继续练习，完成后返回监听模式

---

## 部署和开机自启

### 功能概述

系统支持以下部署功能：

- ✅ **持续监听模式**：程序启动后持续监听唤醒词，无需手动操作
- ✅ **语音唤醒**：说出"voice assistant"或"语音助手"即可唤醒系统
- ✅ **开机自启**：通过systemd服务实现树莓派通电后自动启动
- ✅ **后台运行**：程序在后台运行，不占用终端
- ✅ **自动重启**：程序异常退出时自动重启

### 语音唤醒功能

#### 唤醒词检测器

系统使用轻量级的唤醒词检测器（`asr/wake_word_detector.py`）实现持续监听。

**当前实现**：
- 基于能量检测的简单唤醒词识别
- 支持关键词：`"assistant"`, `"voice assistant"`, `"语音助手"`

**未来改进**（可选）：
- 集成Vosk或Whisper进行更准确的语音识别
- 支持自定义唤醒词训练

#### 唤醒流程

1. 程序启动后，唤醒词检测器开始持续监听
2. 检测到语音能量超过阈值时，尝试匹配唤醒关键词
3. 匹配成功后，触发回调函数，开始练习流程
4. 练习完成后，返回监听模式

### 开机自启动配置

#### 前置条件

1. 确保conda环境已正确安装（参考快速安装部分）
2. 确保程序可以手动正常运行
3. 需要root权限（使用sudo）

#### 安装步骤

**步骤1：检查服务文件**

确保 `oral-assistant.service` 文件存在：
```bash
ls -l oral-assistant.service
```

**步骤2：运行安装脚本**

```bash
sudo ./start_service.sh
```

脚本会自动：
- 复制服务文件到 `/etc/systemd/system/`
- 更新路径配置（conda环境和项目路径）
- 启用服务（开机自启）
- 启动服务

**步骤3：验证服务状态**

```bash
sudo systemctl status oral-assistant
```

应该看到 `active (running)` 状态。

#### 手动安装（可选）

如果自动脚本失败，可以手动安装：

```bash
# 1. 复制服务文件
sudo cp oral-assistant.service /etc/systemd/system/

# 2. 编辑服务文件，更新路径
sudo nano /etc/systemd/system/oral-assistant.service
# 修改以下路径：
# - ExecStart中的conda环境路径
# - WorkingDirectory中的项目路径

# 3. 重新加载systemd
sudo systemctl daemon-reload

# 4. 启用服务
sudo systemctl enable oral-assistant.service

# 5. 启动服务
sudo systemctl start oral-assistant.service
```

#### 服务文件配置说明

`oral-assistant.service` 文件的关键配置：

```ini
[Unit]
Description=树莓派英语口语练习助手
After=network.target sound.target  # 等待网络和音频系统就绪

[Service]
Type=simple
User=pi                            # 运行用户
WorkingDirectory=/home/pi/Desktop/AI_Oral_Assistant
ExecStart=/path/to/conda/envs/oral_assistant/bin/python main.py
Restart=always                     # 异常退出时自动重启
RestartSec=10                      # 重启前等待10秒

[Install]
WantedBy=multi-user.target         # 多用户模式启动
```

**重要路径**：
- `ExecStart`：conda环境中python的完整路径
- `WorkingDirectory`：项目根目录的完整路径

### 服务管理

#### 常用命令

**查看服务状态**：
```bash
sudo systemctl status oral-assistant
```

**查看实时日志**：
```bash
sudo journalctl -u oral-assistant -f
```

**查看最近100行日志**：
```bash
sudo journalctl -u oral-assistant -n 100
```

**停止服务**：
```bash
sudo systemctl stop oral-assistant
```

**重启服务**：
```bash
sudo systemctl restart oral-assistant
```

**禁用开机自启**（但保持服务运行）：
```bash
sudo systemctl disable oral-assistant
```

**启用开机自启**：
```bash
sudo systemctl enable oral-assistant
```

#### 测试服务

1. **重启树莓派**，验证服务是否自动启动：
   ```bash
   sudo reboot
   ```

2. **等待系统启动后**，检查服务状态：
   ```bash
   sudo systemctl status oral-assistant
   ```

3. **查看日志**，确认程序正常运行：
   ```bash
   sudo journalctl -u oral-assistant -n 50
   ```

### 故障排查

#### 服务无法启动

**问题**：`systemctl status` 显示 `failed` 或 `inactive`

**排查步骤**：
1. 查看详细错误日志：
   ```bash
   sudo journalctl -u oral-assistant -n 50 --no-pager
   ```

2. 检查路径是否正确：
   - conda环境路径是否存在
   - 项目目录路径是否正确
   - Python可执行文件路径是否正确

3. 检查权限：
   ```bash
   ls -l /home/pi/Desktop/AI_Oral_Assistant/main.py
   ```

4. 手动测试运行：
   ```bash
   conda activate oral_assistant
   cd /home/pi/Desktop/AI_Oral_Assistant
   python main.py
   ```

#### 唤醒词检测不工作

**问题**：说出唤醒词后没有反应

**排查步骤**：
1. 检查音频设备：
   ```bash
   arecord -l  # 列出录音设备
   ```

2. 检查唤醒词检测器日志：
   ```bash
   sudo journalctl -u oral-assistant | grep -i wake
   ```

3. 测试音频输入：
   ```bash
   arecord -d 5 test.wav  # 录制5秒音频测试
   aplay test.wav         # 播放测试
   ```

4. 如果唤醒词检测失败，可以临时禁用：
   - 编辑服务文件，在 `ExecStart` 后添加 `--no-wake-word`
   - 重启服务

#### 服务频繁重启

**问题**：`systemctl status` 显示服务不断重启

**可能原因**：
1. 程序启动时出错（检查日志）
2. 资源不足（内存或CPU）
3. 依赖服务未就绪

**解决方案**：
1. 查看错误日志找出根本原因
2. 增加 `RestartSec` 延迟时间
3. 在服务文件中添加资源限制：
   ```ini
   MemoryLimit=512M
   CPUQuota=80%
   ```

#### 权限问题

**问题**：服务无法访问音频设备或文件

**解决方案**：
1. 确保用户 `pi` 在 `audio` 组中：
   ```bash
   groups pi  # 查看用户组
   sudo usermod -a -G audio pi  # 添加到audio组
   ```

2. 检查文件权限：
   ```bash
   sudo chown -R pi:pi /home/pi/Desktop/AI_Oral_Assistant
   ```

### 高级配置

#### 自定义唤醒词

编辑 `main.py` 中的唤醒词列表：

```python
self.wake_detector = WakeWordDetector(
    wake_keywords=["your", "custom", "wake", "words"]
)
```

#### 调整资源限制

编辑服务文件，添加资源限制：

```ini
[Service]
MemoryLimit=512M      # 内存限制
CPUQuota=80%         # CPU限制
```

#### 环境变量

如果需要设置环境变量，在服务文件中添加：

```ini
[Service]
Environment="PYTHONPATH=/path/to/project"
Environment="AUDIO_DEVICE=0"
```

#### 卸载服务

如果需要卸载systemd服务：

```bash
# 停止服务
sudo systemctl stop oral-assistant

# 禁用服务
sudo systemctl disable oral-assistant

# 删除服务文件
sudo rm /etc/systemd/system/oral-assistant.service

# 重新加载systemd
sudo systemctl daemon-reload
```

## 项目结构

```
AI_Oral_Assistant/
├── tts/               # 语音合成模块
│   ├── tts_module.py  # TTS核心模块
│   ├── config.py      # 配置文件
│   └── models/        # TTS模型目录
├── asr/               # 语音识别模块
│   ├── raspberry_deploy.py  # 树莓派部署
│   ├── models/        # 音频处理模型
│   └── utils/         # 工具函数
├── scoring/           # 评分模块
│   ├── speech_rater.py      # 评分主类
│   ├── audio_analyzer.py    # 音频分析
│   ├── delivery_scorer.py  # 发音评分
│   └── language_scorer.py  # 内容评分
├── doc/               # 文档和题库
├── main.py            # 主程序入口
├── install.sh         # 安装脚本
├── environment.yml    # Conda环境配置
└── README.md          # 本文件
```

## 配置说明

### 题目配置

题目存储在 `question.md` 文件中，每行一道题。

### 评分权重

可以在 `scoring/config.py` 中调整各指标的权重。

### 音频配置

- 识别模块：16kHz，6通道（可在 `asr/config.py` 中修改）
- 合成模块：22.05kHz，单声道

## 常见问题

### Q1: conda命令未找到

```bash
# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
# 或
source ~/anaconda3/etc/profile.d/conda.sh

# 添加到 ~/.bashrc
echo "source ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
```

### Q2: 环境使用了PyPy而不是CPython

```bash
# 删除环境重新创建
conda env remove -n oral_assistant -y
./install.sh
```

### Q3: TTS模型下载失败

手动下载：
1. 访问 https://github.com/rhasspy/piper/releases
2. 下载 `en_US-amy-medium.onnx` 和 `en_US-amy-medium.onnx.json`
3. 放到 `tts/models/` 目录

### Q4: 音频设备权限问题

```bash
sudo usermod -a -G audio $USER
# 重新登录后生效
```

### Q5: 某些包安装失败

以下包是可选的，不影响基本功能：
- `spacy` - 评分功能会使用简化方法
- `torch` - ASR模块不使用（树莓派不支持）

## 技术说明

### 离线运行

所有模块都支持完全离线运行，不依赖网络连接。

### 模型和算法

- **TTS模块**：使用Piper TTS预训练模型
- **ASR模块**：主要使用纯算法（波束形成、谱减法）
- **评分模块**：大部分功能使用纯算法，可选spaCy增强

### 性能优化

- 适配树莓派ARM架构
- 使用轻量级算法，减少计算量
- 支持流式处理，降低延迟

## 开发计划

- [ ] 集成ASR功能（whisper或vosk）
- [ ] 支持从Word文档读取题库
- [ ] 添加历史记录功能
- [ ] 添加可视化界面

## 许可证

本项目仅供学习和研究使用。

## 参考资料

- Piper TTS: https://github.com/rhasspy/piper
- TOEFL SpeechRater: 见 `doc/rate.txt`
- 树莓派音频配置: https://www.raspberrypi.com/documentation/computers/configuration.html#audio-configuration
