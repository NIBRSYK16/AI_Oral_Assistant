"""
内容准确性评分器
使用千问（Qwen）多模态大模型API评估回答的内容准确性、相关性、逻辑性等
支持音频模态输入，直接分析音频内容
"""
import logging
import os
import base64
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

# 尝试导入DashScope（千问API SDK）
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.warning("DashScope库未安装，内容准确性评分功能将不可用。请安装: pip install dashscope")


class ContentAccuracyRater:
    """
    内容准确性评分器
    使用千问（Qwen）多模态大模型API评估回答是否贴合题目、是否有逻辑性、是否跑题等
    支持直接使用音频模态输入，无需预先转录
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = 'qwen-audio-turbo'):
        """
        初始化内容准确性评分器
        
        Args:
            api_key: DashScope API密钥（如果为None，则从环境变量读取DASHSCOPE_API_KEY）
            model: 使用的模型名称，默认使用 'qwen-audio-turbo'（支持音频模态）
                   可选: 'qwen-audio-turbo', 'qwen-audio', 'qwen-plus', 'qwen-max'
        """
        self.model = model
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        
        if not DASHSCOPE_AVAILABLE:
            logger.warning("DashScope库未安装，内容准确性评分功能不可用")
            self.available = False
            return
        
        if not self.api_key:
            logger.warning("未提供DashScope API密钥，内容准确性评分功能不可用")
            logger.warning("请设置环境变量 DASHSCOPE_API_KEY 或传入 api_key 参数")
            self.available = False
            return
        
        # 设置API密钥
        dashscope.api_key = self.api_key
        self.available = True
        
        logger.info(f"内容准确性评分器初始化完成，使用模型: {self.model}")
    
    def score(self, audio_path: str, question_text: str, 
              recognized_text: Optional[str] = None) -> Dict[str, Any]:
        """
        评估回答的内容准确性
        直接使用音频模态，无需预先转录
        
        Args:
            audio_path: 音频文件路径（支持wav、mp3等格式）
            question_text: 题目文本
            recognized_text: ASR识别的文本（可选，仅用于参考，主要使用音频）
            
        Returns:
            评分结果字典，包含：
            - relevance_score: 相关性评分 (0-1)
            - logic_score: 逻辑性评分 (0-1)
            - completeness_score: 完整性评分 (0-1)
            - overall_score: 总体评分 (0-1)
            - feedback: 详细反馈文本
            - analysis: 详细分析内容
        """
        if not self.available:
            logger.warning("内容准确性评分器不可用")
            return self._get_default_result()
        
        try:
            # 直接使用音频模态调用千问API
            result = self._call_qwen_audio_api(audio_path, question_text, recognized_text)
            return result
            
        except Exception as e:
            logger.error(f"内容准确性评分失败: {e}", exc_info=True)
            return self._get_default_result()
    
    def _build_prompt(self, question_text: str) -> str:
        """
        构建评估提示词
        
        Args:
            question_text: 题目文本
            
        Returns:
            完整的提示词
        """
        prompt = f"""你是一位专业的英语口语评估专家。请仔细分析学生的口语回答，评估其内容质量。

**题目:**
{question_text}

**学生的回答:**（将在音频中提供）

请从以下几个方面评估学生的回答：

1. **相关性 (Relevance)**: 回答是否直接回应了题目？是否切题还是跑题了？
2. **逻辑性和连贯性 (Logic and Coherence)**: 回答的逻辑结构是否清晰？学生的论证能否自圆其说？观点之间的连接是否顺畅？
3. **完整性 (Completeness)**: 回答是否提供了足够的信息？是否充分回答了问题，还是遗漏了重要点？
4. **内容质量 (Content Quality)**: 观点是否清晰、表达是否清楚？内容是否有意义、有实质性？

请为每个方面提供：
- 0.0到1.0之间的分数（1.0为优秀）
- 简短的英文解释

最后，提供：
- 总体分数 (0.0-1.0)
- 英文综合反馈（2-3句话）
- 中文综合反馈（2-3句话）

请严格按照以下JSON格式回复：
{{
    "relevance_score": <float 0.0-1.0>,
    "relevance_feedback": "<brief English explanation>",
    "logic_score": <float 0.0-1.0>,
    "logic_feedback": "<brief English explanation>",
    "completeness_score": <float 0.0-1.0>,
    "completeness_feedback": "<brief English explanation>",
    "content_quality_score": <float 0.0-1.0>,
    "content_quality_feedback": "<brief English explanation>",
    "overall_score": <float 0.0-1.0>,
    "feedback_en": "<comprehensive English feedback>",
    "feedback_zh": "<comprehensive Chinese feedback>"
}}

请只返回JSON，不要包含其他文字。
"""
        return prompt
    
    def _call_qwen_audio_api(self, audio_path: str, question_text: str, 
                             recognized_text: Optional[str] = None) -> Dict[str, Any]:
        """
        调用千问音频模态API进行评估
        
        Args:
            audio_path: 音频文件路径
            question_text: 题目文本
            recognized_text: ASR识别的文本（可选，作为参考）
            
        Returns:
            评估结果字典
        """
        try:
            # 读取音频文件并转换为base64
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # 构建提示词
            prompt = self._build_prompt(question_text)
            
            # 如果有识别文本，可以在提示词中提及（但主要使用音频）
            if recognized_text:
                prompt += f"\n\n参考文本（仅供参考）: {recognized_text}"
            
            # 调用千问多模态API
            # 千问音频模态API支持直接传入音频文件路径或base64编码
            # 这里使用文件路径方式（需要确保文件可访问）
            # 如果使用base64，格式为: "data:audio/wav;base64,{base64_string}"
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "text": "你是一位专业的英语口语评估专家。请仔细分析学生的口语回答，严格按照JSON格式返回评估结果。"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "audio": audio_path  # 直接使用文件路径
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
            
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages
            )
            
            if response.status_code == 200:
                # 解析响应
                result_text = response.output.choices[0].message.content[0].text
                
                # 尝试从响应中提取JSON
                # 千问可能返回带markdown格式的JSON，需要提取
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(0)
                
                result_dict = json.loads(result_text)
                
                # 标准化结果格式
                return {
                    "relevance_score": float(result_dict.get("relevance_score", 0.5)),
                    "logic_score": float(result_dict.get("logic_score", 0.5)),
                    "completeness_score": float(result_dict.get("completeness_score", 0.5)),
                    "content_quality_score": float(result_dict.get("content_quality_score", 0.5)),
                    "overall_score": float(result_dict.get("overall_score", 0.5)),
                    "feedback_en": result_dict.get("feedback_en", ""),
                    "feedback_zh": result_dict.get("feedback_zh", ""),
                    "detailed_feedback": {
                        "relevance": result_dict.get("relevance_feedback", ""),
                        "logic": result_dict.get("logic_feedback", ""),
                        "completeness": result_dict.get("completeness_feedback", ""),
                        "content_quality": result_dict.get("content_quality_feedback", "")
                    }
                }
            else:
                logger.error(f"千问API调用失败: {response.status_code}, {response.message}")
                return self._get_default_result()
            
        except json.JSONDecodeError as e:
            logger.error(f"API响应JSON解析失败: {e}")
            logger.error(f"响应内容: {result_text[:500] if 'result_text' in locals() else 'N/A'}")
            return self._get_default_result()
        except Exception as e:
            logger.error(f"API调用失败: {e}", exc_info=True)
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict[str, Any]:
        """
        返回默认结果（当评分器不可用或出错时）
        
        Returns:
            默认结果字典
        """
        return {
            "relevance_score": 0.5,
            "logic_score": 0.5,
            "completeness_score": 0.5,
            "content_quality_score": 0.5,
            "overall_score": 0.5,
            "feedback_en": "Content accuracy assessment is not available.",
            "feedback_zh": "内容准确性评估不可用。",
            "detailed_feedback": {}
        }


def test_content_accuracy_rater():
    """测试内容准确性评分器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 注意：需要设置DASHSCOPE_API_KEY环境变量
    rater = ContentAccuracyRater()
    
    if not rater.available:
        print("内容准确性评分器不可用（请设置DASHSCOPE_API_KEY环境变量）")
        print("获取API密钥：https://dashscope.console.aliyun.com/")
        return
    
    # 测试数据
    question = "Describe a person who has influenced you the most."
    audio_path = "test_audio.wav"  # 需要实际的音频文件
    
    result = rater.score(audio_path, question)
    
    print("评分结果:")
    print(f"相关性: {result['relevance_score']:.2f}")
    print(f"逻辑性: {result['logic_score']:.2f}")
    print(f"完整性: {result['completeness_score']:.2f}")
    print(f"内容质量: {result['content_quality_score']:.2f}")
    print(f"总体评分: {result['overall_score']:.2f}")
    print(f"\n英文反馈:\n{result['feedback_en']}")
    print(f"\n中文反馈:\n{result['feedback_zh']}")


if __name__ == "__main__":
    test_content_accuracy_rater()
