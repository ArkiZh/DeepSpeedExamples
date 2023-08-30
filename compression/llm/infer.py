

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from transformers.modeling_utils import Conv1D
from deepspeed.compression.helper import convert_conv1d_to_linear

origin_model_name = "gpt2-large"
quantized_weights = "output/W8A8/quant/pytorch_model.bin"

config = AutoConfig.from_pretrained(origin_model_name)

tokenizer = AutoTokenizer.from_pretrained(origin_model_name, use_fast=True)

# model = AutoModelForCausalLM.from_config(config=config)
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
model.to("cuda")


def gen(words):
    inputs = tokenizer.encode(words)
    inputs = torch.tensor([inputs]).to("cuda")
    model.eval()
    import time
    t = []
    with torch.no_grad():
        for i in range(5):
            t1 = time.time()
            outputs = model.generate(inputs)
            t2 = time.time()
            t.append(t2-t1)
            results = tokenizer.decode(outputs[0])
            print(f"=========== Result {i+1}:\n{results}")
    print(f"Time: {t}. Mean: {sum(t)/len(t)}")

text = "Here are 3 countries: "
gen(text)
gen(text)

model = convert_conv1d_to_linear(model, Conv1D)

states = torch.load(quantized_weights,map_location="cuda")


assert isinstance(model, torch.nn.Module)

model.load_state_dict(state_dict=states, strict=True)

gen(text)
gen(text)


print("Done.")

import deepspeed
ds_engine = deepspeed.init_inference(model,
                                 mp_size=2,
                                 dtype=torch.half,
                                 checkpoint=None,
                                 replace_with_kernel_inject=True)
model = ds_engine
gen(text)