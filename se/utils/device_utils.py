"""
设备检测工具模块
检测运行环境是否为树莓派
"""
import platform
import sys


class DeviceDetector:
    @staticmethod
    def is_raspberry_pi():
        """检测是否在树莓派上运行"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    return True
        except:
            pass
        
        # 通过平台信息判断
        system = platform.system()
        if system == "Linux" and "arm" in platform.machine():
            return True
        return False
    
    @staticmethod
    def get_device_info():
        """获取设备信息"""
        info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "is_raspberry_pi": DeviceDetector.is_raspberry_pi()
        }
        return info