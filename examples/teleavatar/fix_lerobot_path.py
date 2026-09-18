#!/usr/bin/env python3
"""
LeRobot 路径修复脚本
在远程服务器上执行: python3 fix_lerobot_path.py

这个脚本会:
1. 检测 lerobot 的实际位置
2. 测试导入是否成功
3. 如果失败，尝试安装缺失的依赖
"""

import sys
import os
import subprocess

def find_lerobot():
    """查找 lerobot 的位置"""
    possible_paths = [
        "/home/lingyu/project/lerobot",
        "/home/lingyu/VLA-project/lerobot",
        "/home/lingyu/anaconda3/envs/lerobot/lib/python3.10/site-packages",
        os.path.expanduser("~/project/lerobot"),
        os.path.expanduser("~/VLA-project/lerobot"),
    ]

    found_paths = []
    for p in possible_paths:
        if os.path.exists(p):
            # 检查是否包含 lerobot 模块
            lerobot_init = os.path.join(p, "lerobot", "__init__.py")
            if os.path.exists(lerobot_init):
                found_paths.append(p)
                print(f"✓ 找到 lerobot: {p}")
            elif os.path.exists(os.path.join(p, "lerobot")):
                found_paths.append(p)
                print(f"✓ 找到 lerobot 目录: {p}")

    return found_paths

def test_import(paths):
    """测试导入"""
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.video_utils import encode_video_frames
        import lerobot
        print(f"\n✓ LeRobot 导入成功!")
        print(f"  版本: {getattr(lerobot, '__version__', 'unknown')}")
        print(f"  位置: {lerobot.__file__}")
        return True
    except ImportError as e:
        print(f"\n✗ LeRobot 导入失败: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    deps = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'datasets': 'datasets',
        'huggingface_hub': 'huggingface-hub',
        'av': 'av',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
    }

    missing = []
    print("\n检查依赖:")
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (需要安装 {package})")
            missing.append(package)

    return missing

def main():
    print("=" * 50)
    print("LeRobot 路径修复工具")
    print("=" * 50)

    print(f"\nPython: {sys.executable}")
    print(f"版本: {sys.version}")

    print("\n查找 lerobot...")
    paths = find_lerobot()

    if not paths:
        print("\n✗ 未找到 lerobot 安装")
        print("请检查 lerobot 是否已安装")
        return 1

    if test_import(paths):
        print("\n" + "=" * 50)
        print("LeRobot 可以正常使用!")
        print("=" * 50)

        # 输出建议的环境变量设置
        print("\n建议添加到 ~/.bashrc 或执行命令前:")
        print(f'export PYTHONPATH="{":".join(paths)}:$PYTHONPATH"')
        return 0

    # 检查缺失的依赖
    missing = check_dependencies()

    if missing:
        print(f"\n缺失依赖: {', '.join(missing)}")
        print("\n尝试安装缺失依赖:")
        print(f"  pip install {' '.join(missing)}")

        # 询问是否安装
        try:
            answer = input("\n是否现在安装? [y/N]: ")
            if answer.lower() == 'y':
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
                print("\n重新测试导入...")
                if test_import(paths):
                    print("✓ 修复成功!")
                    return 0
        except EOFError:
            pass

    print("\n" + "=" * 50)
    print("建议的解决方案:")
    print("=" * 50)
    print("1. 使用 conda 环境中的 Python:")
    print("   /home/lingyu/anaconda3/envs/lerobot/bin/python")
    print("")
    print("2. 或设置 PYTHONPATH:")
    print(f'   export PYTHONPATH="{":".join(paths)}:$PYTHONPATH"')
    print("")
    print("3. 或在 conda 环境中运行:")
    print("   conda activate lerobot")
    print("   python examples/teleavatar/main.py ...")

    return 1

if __name__ == "__main__":
    sys.exit(main())
