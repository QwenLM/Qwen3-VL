import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen3ForCausalLM

# 配置8-bit量化参数
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf4",  # 归一化浮点量化
    bnb_8bit_use_double_quant=True,  # 双重量化优化
    bnb_8bit_quant_storage=torch.uint8
)

# 加载量化模型
model = Qwen3ForCausalLM.from_pretrained(
    "/data/lvm_data_48T/zhangningboo/huggingface_repo/model/Qwen/Qwen3-VL-30B-A3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配设备
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/data/lvm_data_48T/zhangningboo/huggingface_repo/model/Qwen/Qwen3-VL-30B-A3B-Instruct")

# 推理测试
inputs = tokenizer("量子计算的主要挑战是", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))