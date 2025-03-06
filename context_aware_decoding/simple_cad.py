test_token = "<Your_HF_Token>"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from torch.nn import functional as F



def standard_decoding(input_ids, model, tokenizer, max_length=128, temperature=0.5, top_p=0.9):
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    completion_ids = output_ids[:, input_ids.shape[1]:] 
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return completion_ids, output

def context_aware_sampling_fast(input_ids, context_ids, model, tokenizer, alpha=0.9, max_length=128, temperature=0.0):
    generated_tokens = input_ids.clone()
    newline_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
    print_tokens = tokenizer.encode("print", add_special_tokens=False)

    for _ in range(max_length):
        with torch.no_grad():
            # Run both logits in a single batch forward pass
            full_context_outputs = model(generated_tokens)
            full_context_logits = full_context_outputs.logits[:, -1, :]

            question_only_input = generated_tokens[:, len(context_ids):]
            question_only_outputs = model(question_only_input)
            question_only_logits = question_only_outputs.logits[:, -1, :]

        # Efficiently adjust logits
        adjusted_logits = (1 + alpha) * full_context_logits - alpha * question_only_logits

        # Token selection
        if temperature == 0:
            next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(adjusted_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

        # Stop if EOS or newline or print is generated
        if next_token.item() in {tokenizer.eos_token_id} | set(newline_tokens) | set(print_tokens):
            break
    total_prefix_length = input_ids.shape[1]  # Combined input length (context + prompt)
    completion_ids = generated_tokens[:, total_prefix_length:]  # Slicing out input & context 
    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return completion_ids, output

def context_aware_sampling(input_ids, context_ids, model, tokenizer, alpha=0.9, max_length=128, temperature=0.5):
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = test_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token = test_token)


    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model.to(device)
    generated_tokens = input_ids.clone()
    newline_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
    print_tokens = tokenizer.encode("print", add_special_tokens=False)
    for _ in range(max_length):
        with torch.no_grad():
            full_context_outputs = model(generated_tokens)
            full_context_logits = full_context_outputs.logits[:, -1, :] 

            question_only_input = generated_tokens[:, len(context_ids):]
            question_only_outputs = model(question_only_input)
            question_only_logits = question_only_outputs.logits[:, -1, :] 

        adjusted_logits = (1 + alpha) * full_context_logits - alpha * question_only_logits

        if temperature == 0:
            # Use argmax for greedy decoding
            next_token = torch.argmax(adjusted_logits, dim=-1)
            next_token = next_token.view(generated_tokens.shape[0], 1) # reshape for concatenation
        else:
            # Use sampling for non-zero temperature
            probs = F.softmax(adjusted_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

        
        if next_token.item() == tokenizer.eos_token_id or next_token.item() in newline_tokens or next_token.item() in print_tokens:
            break

    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return output



if __name__ == "__main__":
    context = "Write a python function to remove first and last occurrence of a given character from the string. Return only the function, with no other explanation, print statements, or unit tests.\n\n"
    question = "def remove_Occ(s,ch):\n"
    context_input = tokenizer(context, return_tensors="pt").input_ids.to(device)
    question_input = tokenizer(question, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([context_input, question_input], dim=-1)

    model.eval()
    standard_output = standard_decoding(input_ids, model)
    context_aware_output = context_aware_sampling(
                                            input_ids,
                                            context_input,
                                            model,
                                            tokenizer,
                                            alpha=0.9,
                                            max_length=512,
                                            temperature=1.0,
                                        )


    print("__" * 50)
    print("Standard Decoding Output:\n", standard_output)
    print("__" * 50)
    print("Context-Aware Decoding Output:\n", context_aware_output)