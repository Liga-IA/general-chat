#pip install transformers
#pip install accelerate
#https://www.datasciencelearner.com/assertionerror-torch-not-compiled-with-cuda-enabled-fix/
#pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
#O comando acim funcionou para no Windows 11

from transformers import AutoTokenizer, AutoModelForCausalLM, BloomConfig
from accelerate import init_empty_weights
import torch
import os

torch.set_default_tensor_type(torch.cuda.FloatTensor)

config = BloomConfig()

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", 
                                            device_map="auto", torch_dtype="auto",
                                            use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

def chat():
    while True:
        prompt = input(f"You: ")
        os.system("cls")

        input_ids = tokenizer(prompt, return_tensors="pt").to(0)
        print(input_ids)
        sample = model.generate(**input_ids, max_length=500, top_k=1, 
                                temperature=0.7, repetition_penalty = 2.0, min_length=200)

        print("Bot:",tokenizer.decode(sample[0], 
                                    truncate_before_pattern=[r"\n\n^#", "^”'", "\n\n\n"]))

chat()