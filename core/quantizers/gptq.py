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

def is_legal_linear(linear: nn.Module):
    '''check if linear shape is legal to convert fp8'''
    if not isinstance(linear, nn.Linear):
        return False
    '''
    硬件架构的底层优化技术，一般GPU使用128位或者256位内存总线
    一次内存访问可以读取
    - 16个FP8值（128位）
    - 32个FP8值（256位）
    '''
    K, N = linear.weight.data.shape
    return K % 16 == 0 and N % 16 == 0

def find_layers(module: nn.Module, name: str = "", layers=[nn.Linear]) -> {str, nn.Module}:
    
    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, name=name+"."+name1 if name != "" else name1))
    
    return res

def run_gptq_for_sequenial_layers(module: nn.ModuleList):
    for i in range(len(module)):
        # print(type(module[i]))
        sub_layers_dict = find_layers(module[i])
        sub_layers_dict = {name: module for name, module in sub_layers_dict.items() if is_legal_linear(module)}
        for name, module in sub_layers_dict.items():
            print(name, module)
        break

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    qwen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    run_gptq_for_sequenial_layers(qwen_model.model.layers)
    # print(qwen_model)
