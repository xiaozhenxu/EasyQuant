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
        config: GPTQConfig = NULL
    ):
        if not isinstance(layer, nn.Linear):
            print("gptq only support linear now")

        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(
        self,
        input: torch.Tensor,    # (b, s, d) or (s, d)
        output: torch.Tensor
    ):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        tmp = input.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1]))
            input = input.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # input = input.float()
        input = math.sqrt(2 / self.nsamples) * input.float()
        # self.H += 2 / self.nsamples * input.matmul(input.t())
        self.H += input.matmul(input.t())

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

def run_gptq_for_sequenial_layers(module: nn.ModuleList, rotary_emb, input_data: torch.Tensor, gptq_config: GPTQConfig):
    gptqs: dict[str, GPTQ] = dict()
    batch_size, seq_length, hidden_size = input_data.shape

    # create position_ids and attention_mask
    position_ids = torch.arange(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    '''
    shape: [batch_size, 1, seq_length, seq_length]
    下三角矩阵，防止看到未来信息
    attention_mask = [[[[0, -10000, -10000, -10000],
                        [0, 0, -10000, -10000],
                        [0, 0, 0, -10000],
                        [0, 0, 0, 0]]]]
    '''
    # attention_mask = torch.ones((batch_size, seq_length))
    # attention_mask = attention_mask.reshape(batch_size, 1, 1, seq_length)
    # 创建正确的attention_mask（下三角矩阵，1表示可见，0表示被mask）
    attention_mask = torch.tril(torch.ones((batch_size, seq_length, seq_length)))
    attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_length, seq_length]

    device = input_data.device
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    cur_input_tensor = [input_data[n] for n in range(batch_size)]
    for i in range(len(module)):
        # print(type(module[i]))
        decoder_layer = module[i]
        decoder_layer.eval()

        sub_layers_dict = find_layers(decoder_layer)
        sub_layers_dict = {name: module for name, module in sub_layers_dict.items() if is_legal_linear(module)}

        # add gptq for each linear
        for name, layer in sub_layers_dict.items():
            gptqs[name] = GPTQ(layer, gptq_config)
        
        # add forward hooks for each linear
        handles = []
        def add_batch(module_name):
            def add_batch_fn(self, input, output):
                assert len(input) == 1, "shoule have one input tensor"
                gptqs[module_name].add_batch(input[0], output)
            return add_batch_fn
        for name, layer in sub_layers_dict.items():
            handles.append(layer.register_forward_hook(add_batch(name)))

        # iterate data to record stats for gptq
        # next_input_tensor = []
        for j in range(batch_size):
            with torch.no_grad():
                cos, sin = rotary_emb(
                    cur_input_tensor[j].unsqueeze(0), 
                    seq_len=seq_length
                )
                position_embeddings = (cos, sin)
                # output = decoder_layer(
                #     hidden_states=cur_input_tensor[j].unsqueeze(0),
                #     attention_mask=attention_mask[j:j+1],
                #     position_ids=position_ids[j:j+1],
                #     output_attentions=False
                # )
            # next_input_tensor.append(output[0])
        # cur_input_tensor = next_input_tensor


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    qwen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    gptq_config = GPTQConfig()
    data = torch.rand((64, 16, 896))
    run_gptq_for_sequenial_layers(qwen_model.model.layers, qwen_model.model.rotary_emb, data, gptq_config)
    print(qwen_model)
