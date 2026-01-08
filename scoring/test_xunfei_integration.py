"""
讯飞语音评测模块简单的集成测试脚本
用于验证配置是否正确以及能否连接到讯飞服务器
"""
import sys
import os
import wave
import struct

# 将项目根目录添加到python path中，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scoring.config import XUNFEI_CONFIG
from scoring.xunfei_rater import XunfeiRater

def create_dummy_wav(filename):
    """创建一个符合讯飞要求的空白WAV文件 (16k, 16bit, mono)"""
    print(f"创建测试音频文件: {filename}")
    duration = 3.0 # seconds
    sample_rate = 16000
    n_channels = 1
    sampwidth = 2
    n_samples = int(sample_rate * duration)
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        
        # 写入静音数据 (全0)
        # 注意：静音可能会被评测系统拒绝，但至少能验证网络连接
        data = struct.pack('<h', 0) * n_samples
        wf.writeframes(data)

def test_xunfei():
    print("=" * 50)
    print("开始测试讯飞星火语音评测接口")
    print("=" * 50)
    
    # 1. 检查配置
    print("[1/3] 检查配置...")
    if XUNFEI_CONFIG['APPID'] == 'YOUR_APPID':
        print("错误: 请先在 scoring/config.py 中填写你的 APPID, APISecret, APIKey")
        return
    print(f"APPID: {XUNFEI_CONFIG['APPID']}")
    
    # 2. 准备测试文件
    print("[2/3] 准备测试音频...")
    # test_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_xunfei_dummy.wav")
    # create_dummy_wav(test_wav)
    # 使用真实音频测试
    test_wav = os.path.join(project_root, "test_recording.wav")
    if not os.path.exists(test_wav):
        print(f"错误: 找不到测试音频 {test_wav}")
        return
    
    # 评测文本 (与音频内容无关，因为是静音，主要测试流程)
    test_text = "Hello world"
    
    # 3. 发起请求
    print("[3/3] 连接讯飞服务器...")
    rater = XunfeiRater()
    
    try:
        print("正在发送数据...")
        result = rater.score(test_wav, test_text)
        
        if result:
            print("\n测试成功！收到服务器响应:")
            print("-" * 30)
            if 'total_score' in result:
                print(f"总分 (原始): {result.get('total_score')}")
                print(f"总分 (转换后): {result.get('converted_score')}")
            else:
                print("未解析到分数 (可能是静音导致无法评分)")
                
            print(f"原始响应数据长度: {len(result.get('raw_xml', ''))} 字符")
            print(result.get('raw_xml')[:500] + "...") # 打印部分XML
            
        else:
            print("\n测试失败: 未收到结果")
            if rater.error_msg:
                print(f"错误信息: {rater.error_msg}")
            
    except Exception as e:
        print(f"\n发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        pass
        # if os.path.exists(test_wav):
        #     try:
        #         os.remove(test_wav)
        #         print("清理临时文件完成")
        #     except:
        #         pass

if __name__ == "__main__":
    test_xunfei()
