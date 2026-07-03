# OpenVLA 最简推理例程

使用 OpenVLA 模型进行视觉-语言-动作推理的最简示例。

```mermaid
graph LR
    A[📷 图像] --> C[🧠 OpenVLA]
    B[📝 语言指令] --> C
    C --> D[🎮 7-DOF 动作]
```

## 🚀 快速开始

```bash
# 使用随机图片测试
python inference.py

# 使用真实图片
python inference.py --image photo.jpg

# 自定义指令
python inference.py --instruction "pick up the red cube"

# 查看所有参数
python inference.py --help
```

## 📁 文件

| 文件 | 说明 |
|------|------|
| `inference.py` | 最简推理脚本 |
| `requirements.txt` | 依赖记录 |

## 🧠 原理

```
图像 (RGB) + 指令文本
    │
    ▼
PrismaticVLM (SigLIP + DINOv2 视觉编码器 + LLaMA 语言模型)
    │
    ▼
动作解码器 → 7-DOF 动作 [x, y, z, roll, pitch, yaw, gripper]
```

## 📊 预期输出

```
[1/3] 设备: cuda, 数据类型: torch.bfloat16
[2/3] 加载 Processor...
[3/3] 加载 OpenVLA 模型...
✅ 模型加载完成!

📷 使用随机生成的测试图像 (256x256)
📝 指令: "pick up the red cube"

==================================================
  推理结果
==================================================
  动作向量 (7-DOF):
           x: +0.001234
           y: -0.005678
           z: +0.003456
        roll: +0.000123
       pitch: +0.000456
         yaw: -0.000789
     gripper: +0.987654

  耗时: 0.523s
==================================================
```

## 🔧 模型规格

| 模式 | 显存 | 说明 |
|------|------|------|
| BF16 | ~14 GB | 默认，需 Ampere+ GPU |
| 8-bit | ~9 GB | 需 `bitsandbytes` |
| 4-bit | ~6 GB | 需 `bitsandbytes` |

## 📝 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | 随机图片 | 输入图像路径 |
| `--instruction` | "pick up the red cube" | 任务指令 |
| `--model` | "openvla/openvla-7b" | 模型路径 |
| `--fp32` | False | 使用 FP32 (默认 BF16) |
| `--unnorm-key` | "bridge_orig" | 动作反归一化键 |
