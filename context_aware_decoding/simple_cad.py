test_token = "<Your_HF_Token>"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from torch.nn import functional as F

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token = test_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token = test_token)


device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)

def standard_decoding(input_ids, max_length=128, temperature=1.0, top_k=1, top_p=0.9):
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def context_aware_sampling(model, tokenizer, input_ids, context_ids, alpha=0.9, max_length=128, temperature=1.0):
    generated_tokens = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            full_context_outputs = model(generated_tokens)
            full_context_logits = full_context_outputs.logits[:, -1, :] 

            question_only_input = generated_tokens[:, len(context_ids):]
            question_only_outputs = model(question_only_input)
            question_only_logits = question_only_outputs.logits[:, -1, :] 

        adjusted_logits = (1 + alpha) * full_context_logits - alpha * question_only_logits
        adjusted_probs = F.softmax(adjusted_logits / temperature, dim=-1)

        next_token = torch.multinomial(adjusted_probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated_tokens




context = "Here is is the method signature=def remove_Occ(s,ch):"

question = "Output only code for the following question: Write a python function to remove first and last occurrence of a given character from the string."

context_input = tokenizer(context, return_tensors="pt").input_ids.to(device)
question_input = tokenizer(question, return_tensors="pt").input_ids.to(device)

input_ids = torch.cat([context_input, question_input], dim=-1)

model.eval()
standard_output = standard_decoding(input_ids)
output_tokens = context_aware_sampling(
                                        model,
                                        tokenizer,
                                        input_ids,
                                        context_ids=context_input,
                                        alpha=0.5,
                                        max_length=128,
                                        temperature=1.0,
                                    )

context_aware_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("__" * 50)
print("Standard Decoding Output:\n", standard_output)
print("__" * 50)
print("Context-Aware Decoding Output:\n", context_aware_output)