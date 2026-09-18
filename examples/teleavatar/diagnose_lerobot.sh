#!/bin/bash
# LeRobot 诊断和修复脚本
# 在远程服务器上执行: bash diagnose_lerobot.sh

echo "============================================"
echo "LeRobot 环境诊断"
echo "============================================"

echo ""
echo "1. 系统 Python 版本:"
which python3
python3 --version

echo ""
echo "2. 检查 lerobot 可能的位置:"
for path in "/home/lingyu/project/lerobot" "/home/lingyu/VLA-project/lerobot" "/home/lingyu/anaconda3/envs/lerobot/lib/python3.10/site-packages"; do
    if [ -d "$path" ]; then
        echo "   ✓ 存在: $path"
        if [ -d "$path/lerobot" ]; then
            echo "     └─ lerobot 子目录存在"
        fi
    else
        echo "   ✗ 不存在: $path"
    fi
done

echo ""
echo "3. 检查 conda 环境:"
if [ -f "/home/lingyu/anaconda3/etc/profile.d/conda.sh" ]; then
    source /home/lingyu/anaconda3/etc/profile.d/conda.sh
    echo "   Conda 环境列表:"
    conda env list 2>/dev/null || echo "   conda 命令不可用"
fi

echo ""
echo "4. 测试 lerobot 导入 (使用系统 python3):"
python3 << 'EOF'
import sys
print(f"   Python 路径: {sys.executable}")
print(f"   Python 版本: {sys.version}")

# 尝试添加路径并导入
test_paths = [
    "/home/lingyu/project/lerobot",
    "/home/lingyu/VLA-project/lerobot",
    "/home/lingyu/anaconda3/envs/lerobot/lib/python3.10/site-packages",
]

for p in test_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    # Try new path first (lerobot >= 0.5)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import lerobot
    print(f"   ✓ LeRobot 导入成功! (new path: lerobot.common.datasets)")
    print(f"   ✓ LeRobot 位置: {lerobot.__file__}")
except ImportError as e1:
    try:
        # Try old path (lerobot 0.4.x)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import lerobot
        print(f"   ✓ LeRobot 导入成功! (old path: lerobot.datasets)")
        print(f"   ✓ LeRobot 位置: {lerobot.__file__}")
    except ImportError as e2:
        print(f"   ✗ LeRobot 导入失败 (new path): {e1}")
        print(f"   ✗ LeRobot 导入失败 (old path): {e2}")

    # 检查依赖
    print("\n   检查关键依赖:")
    deps = ['torch', 'torchvision', 'datasets', 'huggingface_hub', 'av']
    for dep in deps:
        try:
            __import__(dep)
            print(f"     ✓ {dep}")
        except ImportError:
            print(f"     ✗ {dep} - 缺失")
EOF

echo ""
echo "5. 检查 lerobot 的 Python 版本要求:"
if [ -f "/home/lingyu/project/lerobot/pyproject.toml" ]; then
    echo "   pyproject.toml 中的 Python 版本要求:"
    grep -i "python" /home/lingyu/project/lerobot/pyproject.toml | head -5
fi

echo ""
echo "============================================"
echo "诊断完成"
echo "============================================"
