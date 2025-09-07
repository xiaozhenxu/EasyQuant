import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM

@dataclass
class GPTQConfig:
    bits: int = 4
    damp: int = 0.01
    group_size: int = 64
    block_size: int = 128
    method: str = "per-block"

class GPTQ:
    def __init__(
        self,
        layer: nn.Linear,
        config: GPTQConfig
    ):
        if not isinstance(layer, nn.Linear):
            print("gptq only support linear now")

    def add_batch(
        self,
        input: torch.Tensor,
        output: torch.Tensor
    ):
        pass

    def quantize(
        self
    ):
        pass

def run_gptq_for_sequenial_layer(module: nn.ModuleList):
    pass

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    qwen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
