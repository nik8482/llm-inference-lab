# Part 1: Naive autoregressive inference
# No KV cache - recomputes full attention over all tokens at every step

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class GPT2():
    def __init__(self, model, tokeniser):
        self.model = model
        self.tokeniser = tokeniser

    def generate(self, forward_prompt):
        now = time.time()
        for _ in range(0,300):
            input_ids = tokeniser(forward_prompt, return_tensors = 'pt').to('mps')
            output_logits = model(**input_ids)['logits'][-1][-1]
            output_text = tokeniser.decode(torch.argmax(output_logits))
            forward_prompt = forward_prompt + output_text
        # Decode Tensor ids into words
        later = time.time()
        gen_time = later-now
        tokens_per_sec = 300/gen_time
        return forward_prompt, gen_time, tokens_per_sec
    
def load_model(model_name):
    return AutoTokenizer.from_pretrained(model_name), AutoModelForCausalLM.from_pretrained(model_name).to("mps")

tokeniser, model = load_model("gpt2")
GPT2 = GPT2(model, tokeniser)
sentence,gen_time,tokens_per_sec = GPT2.generate("This is an example of my dogs")
print(sentence)
print(f"Tokens per second: {tokens_per_sec}")
print(f"Generation Time: {gen_time}")