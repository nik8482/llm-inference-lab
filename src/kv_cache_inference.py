# Part 2: Autoregressive inference with KV cache
# Caches key/value tensors to avoid recomputation at each step

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
        for i in range(0,300):
            if i == 0:
                input_ids = tokeniser(forward_prompt, return_tensors = 'pt').to('mps')
                outputs = model(**input_ids, use_cache=True)
                output_logits = outputs['logits'][-1][-1]
                kv_cache = outputs.past_key_values
                next_token = tokeniser.decode(torch.argmax(output_logits))
                sentence = forward_prompt + next_token
            else:
                outputs = model(input_ids=torch.argmax(output_logits).unsqueeze(0).unsqueeze(0).to('mps'), use_cache=True, past_key_values=kv_cache)
                output_logits = outputs['logits'][-1][-1]
                kv_cache = outputs.past_key_values # Recomp it
                next_token = tokeniser.decode(torch.argmax(output_logits))
                sentence = sentence + next_token
        later = time.time()
        gen_time = later-now
        tokens_per_sec = 300/gen_time
        return sentence, gen_time, tokens_per_sec
    
def load_model(model_name):
    return AutoTokenizer.from_pretrained(model_name), AutoModelForCausalLM.from_pretrained(model_name).to("mps")

tokeniser, model = load_model("gpt2")
GPT2 = GPT2(model, tokeniser)
sentence,gen_time,tokens_per_sec = GPT2.generate("This is an example of my dogs")
print(sentence)
print(f"Tokens per second: {tokens_per_sec}")
print(f"Generation Time: {gen_time}")