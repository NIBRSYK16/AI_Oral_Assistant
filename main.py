"""
主程序入口
整合语音合成、语音识别、评分三个模块
实现完整的口语训练助手流程
"""
import os
import sys
import time
import random
import logging
import threading
import shutil
import json
import datetime
import hashlib
from pathlib import Path

# 添加项目根目录到路径（只添加一次，避免模块冲突）
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入各模块（使用包导入）
from tts.tts_module import TextToSpeech
from se.raspberry_deploy import RaspberryPiAudioProcessor
from asr.asr_module import SpeechRecognizer
from scoring.speech_rater import SpeechRater, ScoreResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OralAssistant:
    """口语训练助手主类"""
    
    def __init__(self, enable_wake_word: bool = True):
        """
        初始化助手
        
        Args:
            enable_wake_word: 是否启用唤醒词检测（持续监听模式）
        """
        logger.info("初始化口语训练助手...")
        
        # 初始化各模块
        self.tts = TextToSpeech(auto_load=True)
        self.audio_processor = RaspberryPiAudioProcessor()
        
        # 初始化ASR模块（语音识别）
        try:
            self.asr_recognizer = SpeechRecognizer()
            logger.info("ASR模块初始化成功")
        except Exception as e:
            logger.warning(f"ASR模块初始化失败: {e}，将使用模拟文本")
            self.asr_recognizer = None
        
        self.speech_rater = SpeechRater()
        
        # 唤醒词检测器（可选）
        self.wake_detector = None
        self._wake_detected = False
        self._wake_lock = threading.Lock()
        
        if enable_wake_word:
            try:
                from se.wake_word_detector import WakeWordDetector
                self.wake_detector = WakeWordDetector(
                    wake_keywords=["assistant", "voice assistant", "语音助手"]
                )
                # 设置唤醒回调
                self.wake_detector.set_wake_callback(self._on_wake_detected)
                logger.info("唤醒词检测器已启用")
            except Exception as e:
                logger.warning(f"无法初始化唤醒词检测器: {e}，将使用文本输入模式")
                self.wake_detector = None
        
        # 题库路径
        self.question_bank_path = os.path.join(os.path.dirname(__file__), "doc", "口语题库.docx")
        self.questions = self._load_questions()
        
        # 音频缓存配置
        self.cache_dir = os.path.join(project_root, "assets", "audio_cache")
        self.fixed_phrases = {
            "welcome": "Hello, I'm ready to help you practice English.",
            "instruction": "You have 15 seconds to prepare and 45 seconds to answer.",
            "start_answer": "Please start your answer now.",
            "goodbye": "Goodbye! Keep practicing!"
        }
        self._init_audio_cache()
        
        # 录音文件路径
        self.recorded_audio_path = "recorded_response.wav"
        
        # ASR置信度缓存
        self._last_asr_confidence = None
        
        # 运行模式
        self.continuous_mode = enable_wake_word and self.wake_detector is not None
        
        logger.info("口语训练助手初始化完成")
    
    def _on_wake_detected(self, keyword: str):
        """唤醒词检测回调"""
        with self._wake_lock:
            self._wake_detected = True
        logger.info(f"唤醒词检测回调: {keyword}")

    def _init_audio_cache(self):
        """初始化音频缓存，检查更新"""
        logger.info("正在检查音频缓存...")
        print("正在检查语音资源...")
        
        # 1. 确保目录结构
        fixed_dir = os.path.join(self.cache_dir, "fixed")
        questions_dir = os.path.join(self.cache_dir, "questions")
        os.makedirs(fixed_dir, exist_ok=True)
        os.makedirs(questions_dir, exist_ok=True)
        
        # 2. 处理固定短语
        for key, text in self.fixed_phrases.items():
            path = os.path.join(fixed_dir, f"{key}.wav")
            if not os.path.exists(path):
                logger.info(f"生成固定音频: {key}")
                self.tts.synthesize_to_file(text, path)
                
        # 3. 处理题库
        current_hashes = set()
        updated_count = 0
        
        # 扫描现有题目并生成音频
        for question in self.questions:
            # 使用MD5生成文件名
            q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            current_hashes.add(q_hash)
            
            path = os.path.join(questions_dir, f"{q_hash}.wav")
            if not os.path.exists(path):
                logger.info(f"生成新题目音频: {question[:20]}...")
                print(f"正在生成题目音频: {question[:30]}...")
                self.tts.synthesize_to_file(question, path)
                updated_count += 1
                
        # 清理已删除题目的音频
        removed_count = 0
        for filename in os.listdir(questions_dir):
            if filename.endswith('.wav'):
                file_hash = filename[:-4]
                if file_hash not in current_hashes:
                    os.remove(os.path.join(questions_dir, filename))
                    removed_count += 1
                    
        if updated_count > 0 or removed_count > 0:
            msg = f"资源更新完成: 新增 {updated_count} 条，删除 {removed_count} 条"
            logger.info(msg)
            print(msg)
        else:
            logger.info("语音资源已就绪")

    def _play_cached(self, key: str, subdir: str = "fixed"):
        """播放缓存音频"""
        path = os.path.join(self.cache_dir, subdir, f"{key}.wav")
        if os.path.exists(path):
            self.tts.play_wav_file(path, blocking=True)
        else:
            logger.warning(f"缓存音频丢失: {path}")
            # 降级：如果找不到文件，尝试实时合成（如果是固定短语）
            if subdir == "fixed" and key in self.fixed_phrases:
                self.tts.speak(self.fixed_phrases[key], blocking=True)

    
    def _load_questions(self) -> list:
        """
        加载题库
        从question.md文件读取题目
        """
        question_file = os.path.join(os.path.dirname(__file__), "question.md")
        questions = []
        
        # 尝试从文件读取
        if os.path.exists(question_file):
            try:
                with open(question_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            questions.append(line)
            except Exception as e:
                logger.warning(f"读取题库文件失败: {e}")
        
        # 如果读取失败或文件不存在，使用默认题目
        if not questions:
            questions = [
                "Describe a person who has influenced you the most.",
                "What is your favorite place to visit? Why?",
                "Describe a memorable event in your life.",
                "What is the most important quality for a friend?",
                "Describe a book or movie that impressed you.",
                "What is your ideal job? Why?",
                "Describe a skill you want to learn.",
                "What is the best way to relax?",
            ]
        
        logger.info(f"加载了 {len(questions)} 道题目")
        return questions
    
    def _wake_up(self) -> bool:
        """
        等待用户唤醒
        检测包含"语音助手"相关的语句来唤醒
        """
        logger.info("等待用户唤醒...")
        print("\n" + "="*50)
        print("口语训练助手已启动")
        print("="*50)
        
        wake_keywords = ["语音助手", "voice assistant", "assistant"]
        
        # 如果启用了唤醒词检测器，启动持续监听
        if self.wake_detector:
            print("\n提示：说出唤醒词来启动练习（例如：'voice assistant' 或 '语音助手'）")
            print("或者输入 'start' 开始练习，输入 'quit' 退出")
            
            wake_listening_started = False
            try:
                # 启动唤醒词检测器
                self.wake_detector.start_listening()
                wake_listening_started = True
                print("正在监听唤醒词...")
                
                # 重置唤醒标志
                with self._wake_lock:
                    self._wake_detected = False
                
                # 持续监听，直到检测到唤醒词或用户输入
                while True:
                    # 检查是否检测到唤醒词
                    with self._wake_lock:
                        if self._wake_detected:
                            self._wake_detected = False
                            logger.info("检测到唤醒词，开始练习")
                            print("\n✓ 检测到唤醒词！")
                            self._play_cached("welcome")
                            return True
                    
                    # 非阻塞检查用户输入（使用select或timeout）
                    import select
                    import sys
                    
                    # 检查是否有输入（非阻塞）
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        user_input = input().strip()
                        
                        if user_input.lower() in ['start', 's']:
                            logger.info("用户直接开始")
                            return True
                        elif user_input.lower() in ['quit', 'q', 'exit']:
                            logger.info("用户退出")
                            return False
                    
                    time.sleep(0.1)  # 短暂休眠，避免CPU占用过高
                    
            except KeyboardInterrupt:
                logger.info("用户中断")
                return False
            except Exception as e:
                logger.error(f"唤醒检测出错: {e}")
                print(f"\n警告: 唤醒词检测启动失败 ({e})")
                print("已自动切换到文本输入模式")
                # 回退到文本输入模式，继续等待用户输入
                wake_listening_started = False
            finally:
                # 停止唤醒词检测器（如果已启动）
                if wake_listening_started and self.wake_detector:
                    try:
                        self.wake_detector.stop_listening()
                    except Exception as e:
                        logger.warning(f"停止唤醒词检测器时出错: {e}")
            
            # 如果唤醒词检测失败，回退到文本输入模式
            if not wake_listening_started:
                print("\n提示：输入 'start' 开始练习，输入 'quit' 退出")
                
                while True:
                    try:
                        user_input = input("\n输入命令: ").strip()
                        
                        if not user_input:
                            continue
                        
                        user_input_lower = user_input.lower()
                        
                        # 检测唤醒关键词
                        if any(keyword.lower() in user_input_lower for keyword in wake_keywords):
                            logger.info(f"文本唤醒: {user_input}")
                            self._play_cached("welcome")
                            return True
                        elif user_input.lower() in ['start', 's']:
                            logger.info("用户直接开始")
                            return True
                        elif user_input.lower() in ['quit', 'q', 'exit']:
                            logger.info("用户退出")
                            return False
                        else:
                            print("未检测到唤醒词，请输入 'start' 开始")
                            
                    except KeyboardInterrupt:
                        logger.info("用户中断")
                        return False
                    except Exception as e:
                        logger.error(f"输入处理出错: {e}")
                        continue
        else:
            # 文本输入模式（备用方案）
            print("\n提示：输入 'start' 开始练习，输入 'quit' 退出")
            
            while True:
                try:
                    user_input = input("\n输入命令: ").strip()
                    
                    if not user_input:
                        continue
                    
                    user_input_lower = user_input.lower()
                    
                    # 检测唤醒关键词
                    if any(keyword.lower() in user_input_lower for keyword in wake_keywords):
                        logger.info(f"文本唤醒: {user_input}")
                        self.tts.speak("Hello, I'm ready to help you practice English.", blocking=True)
                        return True
                    elif user_input.lower() in ['start', 's']:
                        logger.info("用户直接开始")
                        return True
                    elif user_input.lower() in ['quit', 'q', 'exit']:
                        logger.info("用户退出")
                        return False
                    else:
                        print("未检测到唤醒词，请输入 'start' 开始")
                        
                except KeyboardInterrupt:
                    logger.info("用户中断")
                    return False
                except Exception as e:
                    logger.error(f"输入处理出错: {e}")
                    continue
    
    def _listen_for_wake_word(self) -> str:
        """
        监听唤醒词
        如果唤醒词检测器可用，则进行语音识别
        否则返回None，使用文本输入
        
        Returns:
            识别到的文本，如果未识别或检测器不可用则返回None
        """
        try:
            # 使用唤醒词检测器
            if hasattr(self, 'wake_detector') and self.wake_detector:
                # 唤醒词检测器在后台持续监听，通过回调函数通知
                # 这里返回None，实际唤醒通过回调处理
                return None
            else:
                return None
            
        except Exception as e:
            logger.debug(f"语音唤醒检测不可用: {e}")
            return None
    
    def _select_question(self) -> str:
        """随机选择题目"""
        if not self.questions:
            return "Please describe your favorite place."
        return random.choice(self.questions)
    
    def _present_question(self, question: str):
        """
        出题：播放缓存的题目音频
        """
        logger.info(f"出题: {question}")
        
        # 播放题目
        print(f"\n题目: {question}")
        
        # 1. 播放题目音频
        q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        self._play_cached(q_hash, subdir="questions")
        
        # 2. 播放固定指令
        self._play_cached("instruction")
    
    def _wait_preparation(self, duration: int = 15):
        """
        等待准备时间
        """
        logger.info(f"等待准备时间: {duration}秒")
        print(f"\n准备时间: {duration}秒")
        
        # 倒计时
        for i in range(duration, 0, -1):
            print(f"剩余时间: {i}秒", end='\r')
            time.sleep(1)
        
        print("\n准备时间结束！")
        
        # 提示开始答题
        self._play_cached("start_answer")
    
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
            # 录音
            # 注意：这里需要修改识别模块以支持保存音频和返回文本
            # 暂时使用简化实现
            
            # 启动录音
            self.audio_processor.start_recording(duration=duration)
            
            # 等待录音完成
            while self.audio_processor.is_recording:
                time.sleep(0.1)
            
            # 等待处理完成
            while self.audio_processor.is_processing:
                time.sleep(0.1)
            
            # 获取处理后的音频路径
            audio_path = self.audio_processor.output_audio_path
            
            if not os.path.exists(audio_path):
                logger.warning(f"音频文件不存在: {audio_path}，尝试使用默认路径")
                audio_path = self.recorded_audio_path
                # 如果默认路径也不存在，创建一个空文件提示
                if not os.path.exists(audio_path):
                    logger.error("音频文件不存在，请检查录音模块")
                    return "", ""
            
            # 调用ASR识别
            recognized_text, asr_confidence = self._recognize_speech(audio_path)
            
            logger.info(f"识别结果: {recognized_text}")
            if asr_confidence is not None:
                logger.info(f"ASR置信度: {asr_confidence:.2f}")
            print(f"\n识别结果: {recognized_text}")
            
            # 保存置信度供评分使用
            self._last_asr_confidence = asr_confidence
            
            return recognized_text, audio_path
            
        except Exception as e:
            logger.error(f"录音失败: {e}")
            print(f"录音失败: {e}")
            return "", ""
    
    def _recognize_speech(self, audio_path: str) -> tuple:
        """
        语音识别
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (recognized_text, confidence) 元组
        """
        if self.asr_recognizer is None or not self.asr_recognizer.is_available():
            logger.warning("ASR模块不可用，使用模拟文本")
            return "I think the most important quality for a friend is honesty. A good friend should be trustworthy and reliable.", None
        
        try:
            # 使用ASR模块进行识别
            text, confidence = self.asr_recognizer.recognize_with_confidence(audio_path)
            
            if not text or len(text.strip()) == 0:
                logger.warning("ASR识别结果为空，使用模拟文本")
                return "I think the most important quality for a friend is honesty. A good friend should be trustworthy and reliable.", None
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"ASR识别失败: {e}")
            # 返回模拟文本作为备选
            return "I think the most important quality for a friend is honesty. A good friend should be trustworthy and reliable.", None
    
    def _score_response(self, audio_path: str, text: str, question_text: str = None) -> ScoreResult:
        """
        评分
        
        Args:
            audio_path: 音频文件路径
            text: 识别文本
            question_text: 题目文本(用于讯飞topic模式)
            
        Returns:
            评分结果
        """
        logger.info("开始评分...")
        print("\n正在评分，请稍候...")
        
        try:
            # 获取ASR置信度（如果可用）
            asr_confidence = getattr(self, '_last_asr_confidence', None)
            
            result = self.speech_rater.score(
                audio_path=audio_path,
                text=text,
                asr_confidence=asr_confidence,  # 传入ASR置信度
                task_type="independent",
                reference_text=question_text
            )
            
            return result
            
        except Exception as e:
            logger.error(f"评分失败: {e}")
            print(f"评分失败: {e}")
            raise
    
    def _present_feedback(self, result: ScoreResult):
        """
        播放评价（中英文）
        """
        logger.info("播放评价...")
        
        print("\n" + "="*50)
        print("评分结果")
        print("="*50)
        
        # 如果有ASR识别文本，打印出来
        if hasattr(result, 'recognized_text') and result.recognized_text:
            print(f"ASR识别内容 (修正后): {result.recognized_text}\n")
            
        print(f"总分: {result.raw_score:.2f} / 4.0")
        print(f"发音分: {result.delivery_score:.2f} / 4.0")
        print(f"内容分: {result.language_score:.2f} / 4.0")
        print("\n评价（英文）:")
        print(result.feedback_en)
        print("\n评价（中文）:")
        print(result.feedback_zh)
        print("="*50)
        
        # 播放英文评价
        print("\n播放英文评价...")
        self.tts.speak(result.feedback_en, blocking=True)
        
        # 播放中文评价
        print("\n播放中文评价...")
        # 注意：如果TTS不支持中文，可以跳过或使用其他方法
        # self.tts.speak(result.feedback_zh, blocking=True)
    
    def _ask_continue(self) -> bool:
        """
        询问是否继续
        """
        print("\n是否继续练习？(y/n): ", end='')
        response = input().strip().lower()
        return response == 'y' or response == 'yes'
    
    def _say_goodbye(self):
        """说再见"""
        self._play_cached("goodbye")
        print("\n再见！继续加油练习！")
    
    def _save_history(self, question: str, audio_path: str, text: str, result: ScoreResult):
        """
        保存历史记录
        将录音和评价结果保存到 history/timestamp 目录下
        """
        try:
            # 生成时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建目录结构
            history_root = os.path.join(project_root, "history")
            session_dir = os.path.join(history_root, timestamp)
            os.makedirs(session_dir, exist_ok=True)
            
            # 1. 复制音频文件
            target_audio_path = os.path.join(session_dir, "response.wav")
            if os.path.exists(audio_path):
                shutil.copy2(audio_path, target_audio_path)
            else:
                logger.warning(f"音频文件不存在，无法保存: {audio_path}")
            
            # 2. 保存完整报告 (JSON)
            report_data = {
                "timestamp": timestamp,
                "question": question,
                "recognized_text": text,
                "score_result": result.to_dict()
            }
            
            report_path = os.path.join(session_dir, "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
                
            # 3. 保存简易文本报告 (TXT)
            txt_path = os.path.join(session_dir, "report.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"练习时间: {timestamp}\n")
                f.write(f"题目: {question}\n")
                f.write(f"识别回答: {text}\n")
                f.write("-" * 30 + "\n")
                f.write(f"总分: {result.raw_score:.2f} / 4.0\n")  # 分数已经是1.5-4区间
                f.write(f"发音分: {result.delivery_score:.2f} / 4.0\n")
                f.write(f"内容分: {result.language_score:.2f} / 4.0\n")
                f.write("-" * 30 + "\n")
                f.write("评价 (En):\n")
                f.write(f"{result.feedback_en}\n\n")
                f.write("评价 (Zh):\n")
                f.write(f"{result.feedback_zh}\n")
            
            logger.info(f"历史记录已保存至: {session_dir}")
            print(f"本次练习记录已保存")
            
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")
            print(f"保存历史记录失败: {e}")

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
                    result = self._score_response(audio_path, text, question_text=question)
                    
                    # 关键修改：如果评分过程提取了ASR识别文本，更新用于记录的text
                    if hasattr(result, 'recognized_text') and result.recognized_text:
                        text = result.recognized_text
                        logger.info(f"更新记录文本为ASR识别结果: {text[:20]}...")

                    # 6. 播放评价
                    self._present_feedback(result)
                    
                    # 保存历史记录
                    self._save_history(question, audio_path, text, result)
                    
                except Exception as e:
                    logger.error(f"评分过程出错: {e}")
                    print(f"评分失败: {e}")
                
                # 7. 询问是否继续
                if not self._ask_continue():
                    break
            
            # 8. 说再见
            self._say_goodbye()
            
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            print("\n\n程序已中断")
            self._say_goodbye()
        except Exception as e:
            logger.error(f"程序运行出错: {e}", exc_info=True)
            print(f"\n程序出错: {e}")
        finally:
            # 清理资源
            if self.wake_detector:
                self.wake_detector.cleanup()
            self.audio_processor.cleanup()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='树莓派英语口语练习助手')
    parser.add_argument('--no-wake-word', action='store_true', 
                       help='禁用唤醒词检测，使用文本输入模式')
    parser.add_argument('--daemon', action='store_true',
                       help='后台运行模式（用于systemd服务）')
    
    args = parser.parse_args()
    
    print("="*50)
    print("基于树莓派的英语口语训练助手")
    print("="*50)
    print("\n系统初始化中...")
    
    try:
        # 根据参数决定是否启用唤醒词
        enable_wake = not args.no_wake_word
        assistant = OralAssistant(enable_wake_word=enable_wake)
        assistant.run()
    except Exception as e:
        logger.error(f"初始化失败: {e}", exc_info=True)
        print(f"\n初始化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

