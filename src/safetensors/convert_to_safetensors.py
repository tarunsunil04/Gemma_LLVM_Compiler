from transformers import AutoModelForCausalLM
import torch

#A python script to download the model as is, as safetensors. This could be modified, to provide quantization support, to improve it much further.  

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.bfloat16,)
save_directory = "./model_dir"

model.save_pretrained(save_directory, safe_serialization=True)