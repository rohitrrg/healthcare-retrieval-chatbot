import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.schema.runnable import Runnable
import os

hf_token = os.getenv("HF_TOKEN")

class LLM(Runnable):
    def __init__(self):
        self.ckpt = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt, token=hf_token)

        bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.ckpt,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder=os.environ.get("HF_HOME", "/tmp/offload"),
            token=hf_token
        )
        print(f'✅ LLM {self.ckpt} loaded on GPU.')
    
    def invoke(self, prompt, config=None):
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])