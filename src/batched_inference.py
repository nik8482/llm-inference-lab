# Part 3: Batched inference
# Handle multiple requests simultaneously, measure throughput vs batch size
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

    def generate_batch(self, batch):
        tokeniser.padding_side = "left"
        tokeniser.pad_token = tokeniser.eos_token
        
        now=time.time()
        # 1. Initial Pass
        inputs = tokeniser(batch, return_tensors='pt', padding=True).to('mps')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        
        # Get logits for the LAST token of every sentence in the batch: (BatchSize, VocabSize)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1)
        
        kv_cache = outputs.past_key_values
        
        # Store all generated tokens in a list of lists or a tensor
        all_generated_ids = [next_token_ids.unsqueeze(-1)]

        # 2. Iterative Generation
        for i in range(1, 300):
            # We only pass the SINGLE newest token per batch element
            current_input_id = all_generated_ids[-1] 
            
            # We must update the attention mask to account for the new token
            new_mask_bit = torch.ones((current_input_id.shape[0], 1), dtype=torch.long, device='mps')
            attention_mask = torch.cat([attention_mask, new_mask_bit], dim=-1)

            outputs = model(
                input_ids=current_input_id,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            kv_cache = outputs.past_key_values
            all_generated_ids.append(next_token_ids)

        # 3. Final Decoding
        later = time.time()
        full_ids = torch.cat([input_ids] + all_generated_ids, dim=-1)
        decoded_sentences = tokeniser.batch_decode(full_ids, skip_special_tokens=True)
        gen_time = later-now
        tokens_per_sec = 300*len(batch)/gen_time
        return decoded_sentences, gen_time, tokens_per_sec
 
def load_model(model_name):
    return AutoTokenizer.from_pretrained(model_name), AutoModelForCausalLM.from_pretrained(model_name).to("mps")

tokeniser, model = load_model("gpt2")
GPT2 = GPT2(model, tokeniser)
sentences,gen_time,tokens_per_sec = GPT2.generate_batch(["This is an example of my dogs","London is a city of", "My favourite food is"])
print(sentences)
print(f"Tokens per second: {tokens_per_sec}")
print(f"Generation Time: {gen_time}")