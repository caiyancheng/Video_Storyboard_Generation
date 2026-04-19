"""
单卡单图推理脚本：给定一张图片，用 vidi 生成故事板 prompt。

用法（服务器上，单卡）：
  python run_storyboard_inference.py --image /path/to/frame.png
  python run_storyboard_inference.py --image /path/to/frame.png --output result.txt

批量（指定一个目录，处理所有 .png）：
  python run_storyboard_inference.py --image_dir /path/to/frames/ --output_dir /path/to/out/
"""

import os
import argparse
from pathlib import Path

import torch
from dattn import get_dattn_cls
from vllm import LLM, SamplingParams

from storyboard_prompt_template import STORYBOARD_SYSTEM_PROMPT

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL_PATH = "/mnt/bn/yilin4/yancheng/vidi_edit_ckpt/dattn_gemma3-12b_pt_30k_interleave_nos"
VLLM_MAX_LENGTH = 72000
WARMUP_LENGTHS = [int(3e4), int(4e4)]


# ── Encoder ──────────────────────────────────────────────────────────────────

class MMEncoder:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        model_cls = get_dattn_cls(model_path)
        self.model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            mm_vision_tower="google/siglip2-so400m-patch14-384",
            mm_audio_tower="openai/whisper-large-v3",
            mm_splits=4,
            attn_implementation="decomposed_attention",
            torch_dtype=torch.bfloat16,
        ).to(device, dtype=torch.bfloat16)
        self.model.eval()
        self.model.config.use_cache = True
        self.processor = self.model.mm_processor

    @torch.inference_mode()
    def encode(self, image_path: str):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": STORYBOARD_SYSTEM_PROMPT},
                ],
            },
            {"role": "assistant", "content": ""},
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            load_audio_from_video=True,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            fps=1.0,
            padding=True,
            video_load_backend="decord",
            enable_thinking=False,
            return_assistant_tokens_mask=False,
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device, dtype=torch.bool)
        inputs["images"] = inputs["images"].to(self.device, dtype=torch.bfloat16) if inputs.get("images") is not None else None
        inputs["videos"] = inputs["videos"].to(self.device, dtype=torch.bfloat16) if inputs.get("videos") is not None else None
        inputs["audios"] = inputs["audios"].to(self.device, dtype=torch.bfloat16) if inputs.get("audios") is not None else None
        inputs["image_sizes"] = [inputs["image_sizes"].cpu().numpy()] if inputs.get("images") is not None else []
        inputs["video_sizes"] = [inputs["video_sizes"].cpu().numpy()] if inputs.get("videos") is not None else []
        inputs["audio_sizes"] = [inputs["audio_sizes"].cpu().numpy()] if inputs.get("audios") is not None else []

        (_, _, attention_mask, _, prompt_embeds, _) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=inputs["input_ids"],
            position_ids=None,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            labels=None,
            images=inputs["images"],
            image_sizes=inputs["image_sizes"],
            image_padding_mask=None,
            videos=inputs["videos"],
            video_sizes=inputs["video_sizes"],
            video_padding_mask=inputs.get("video_padding_mask"),
            audios=inputs["audios"],
            audio_sizes=inputs["audio_sizes"],
            audio_padding_mask=None,
        )
        return prompt_embeds.squeeze(0), attention_mask.squeeze(0)


# ── Runner ────────────────────────────────────────────────────────────────────

class VllmRunner:
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            enable_prompt_embeds=True,
            gpu_memory_utilization=0.5,
            attention_backend="FLEX_ATTENTION",
            enable_prefix_caching=False,
            enforce_eager=False,
            max_model_len=VLLM_MAX_LENGTH,
            enable_chunked_prefill=False,
            disable_hybrid_kv_cache_manager=True,
        )

    def warmup(self, embed_dim: int):
        print("=== Warmup ===")
        for length in WARMUP_LENGTHS:
            embeds = torch.randn(length, embed_dim).cuda()
            mask = torch.ones(length).cuda()
            mask[: int(length * 0.25)] = 2
            mask[int(length * 0.25): int(length * 0.5)] = 4
            mask[int(length * 0.5):] = 3
            mask[-100:] = 10
            self._generate_one(embeds, mask, max_tokens=4)
        print("=== Warmup Done ===")

    def _generate_one(self, embeds, mask, max_tokens, temperature=0.0):
        outputs = self.llm.generate(
            prompts=[{"prompt_embeds": embeds, "vidi3_attention_mask": mask}],
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text

    def generate(self, embeds, mask, max_tokens=8192, temperature=0.0) -> str:
        return self._generate_one(embeds, mask, max_tokens, temperature)


# ── Main ──────────────────────────────────────────────────────────────────────

def infer_one(encoder: MMEncoder, runner: VllmRunner, image_path: str, max_tokens: int, temperature: float) -> str:
    embeds, mask = encoder.encode(image_path)
    return runner.generate(embeds, mask, max_tokens=max_tokens, temperature=temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--output", type=str, default=None, help="Output .txt path (single mode)")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of .png files (batch mode)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (batch mode)")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_warmup", action="store_true")
    args = parser.parse_args()

    encoder = MMEncoder(args.model_path)
    runner = VllmRunner(args.model_path)

    if not args.no_warmup:
        runner.warmup(embed_dim=encoder.model.config.hidden_size)

    if args.image:
        result = infer_one(encoder, runner, args.image, args.max_tokens, args.temperature)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            print(f"Saved -> {args.output}")
        else:
            print(result)

    elif args.image_dir:
        assert args.output_dir, "--output_dir required in batch mode"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        images = sorted(Path(args.image_dir).glob("*.png"))
        print(f"Found {len(images)} images")
        for img in images:
            out_path = Path(args.output_dir) / img.with_suffix(".txt").name
            if out_path.exists():
                continue
            result = infer_one(encoder, runner, str(img), args.max_tokens, args.temperature)
            out_path.write_text(result, encoding="utf-8")
            print(f"  ✓ {img.name}")
    else:
        parser.error("Provide --image or --image_dir")


if __name__ == "__main__":
    main()
