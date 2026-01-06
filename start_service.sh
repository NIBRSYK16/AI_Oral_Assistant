#!/bin/bash
# 安装和启动systemd服务脚本

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_NAME="oral-assistant"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "=========================================="
echo "安装口语练习助手系统服务"
echo "=========================================="
echo ""

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then 
    echo "错误: 请使用sudo运行此脚本"
    echo "用法: sudo ./start_service.sh"
    exit 1
fi

# 检查服务文件是否存在
if [ ! -f "$SERVICE_FILE" ]; then
    echo "错误: 服务文件不存在: $SERVICE_FILE"
    exit 1
fi

# 更新服务文件中的路径（如果需要）
CONDA_ENV_PATH=$(conda info --base 2>/dev/null || echo "/home/pi/miniconda3")
PROJECT_DIR="$SCRIPT_DIR"

echo "配置信息:"
echo "  - Conda环境路径: ${CONDA_ENV_PATH}"
echo "  - 项目目录: ${PROJECT_DIR}"
echo ""

# 复制服务文件
echo "步骤 1/4: 复制服务文件..."
cp "$SERVICE_FILE" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"

# 更新服务文件中的路径
sed -i "s|/home/pi/miniconda3|${CONDA_ENV_PATH}|g" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"
sed -i "s|/home/pi/Desktop/AI_Oral_Assistant|${PROJECT_DIR}|g" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"

echo "  ✓ 服务文件已复制到 ${SYSTEMD_DIR}"

# 重新加载systemd
echo ""
echo "步骤 2/4: 重新加载systemd..."
systemctl daemon-reload
echo "  ✓ systemd已重新加载"

# 启用服务（开机自启）
echo ""
echo "步骤 3/4: 启用服务（开机自启）..."
systemctl enable "${SERVICE_NAME}.service"
echo "  ✓ 服务已启用，将在开机时自动启动"

# 启动服务
echo ""
echo "步骤 4/4: 启动服务..."
systemctl start "${SERVICE_NAME}.service"
echo "  ✓ 服务已启动"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "服务管理命令:"
echo "  查看状态:  sudo systemctl status ${SERVICE_NAME}"
echo "  查看日志:  sudo journalctl -u ${SERVICE_NAME} -f"
echo "  停止服务:  sudo systemctl stop ${SERVICE_NAME}"
echo "  重启服务:  sudo systemctl restart ${SERVICE_NAME}"
echo "  禁用自启:  sudo systemctl disable ${SERVICE_NAME}"
echo ""

