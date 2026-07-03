import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig,AutoConfig

MODEL_PATH = os.path.abspath("./openvla-7b")
UNNORM_KEY = "bridge_orig"


def build_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def load_local_4bit_model():
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，无法进行 4bit GPU 部署")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model path (local): {MODEL_PATH}")

    # 1. 正常加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # 注意：这里去掉了 use_double_quant，保持和 Issue 成功案例完全一致
    )

    # 官方 Issue 验证通过的加载写法
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",  # 必须写死显卡号，绝不能写 "auto" 或 {"": 0}
        trust_remote_code=True
        # 不要加 low_cpu_mem_usage，否则也会触发内部检测
    )

    return model, processor


@torch.inference_mode()
def predict_action(model, processor, image: Image.Image, instruction: str):
    prompt = build_prompt(instruction)
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    return action.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--instruction", default="pick up the red cube")
    args = parser.parse_args()

    model, processor = load_local_4bit_model()

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    t0 = time.time()
    action = predict_action(model, processor, image, args.instruction)
    print("action:", action)
    print(f"latency: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
