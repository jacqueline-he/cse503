test_token = "<Your_HF_Token>"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from torch.nn import functional as F
import pdb


def standard_decoding(input_ids, model, tokenizer, max_length=128, temperature=0.5):
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
    )
    completion_ids = output_ids[:, input_ids.shape[1]:] 
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return completion_ids, output

# def context_aware_sampling_fast(input_ids, context_ids, model, tokenizer, alpha=0.9, max_length=128, temperature=0.0):
#     batch_size, seq_len = input_ids.shape
#     device = model.device
#     generated_tokens = input_ids.clone()

#     newline_tokens = set(tokenizer.encode("\n\n", add_special_tokens=False))
#     print_tokens = set(tokenizer.encode("print", add_special_tokens=False))
#     eos_token = {tokenizer.eos_token_id}

#     # Pre-allocate buffer
#     buffer = torch.full((batch_size, seq_len + max_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
#     buffer[:, :seq_len] = input_ids  # Copy initial tokens
#     current_length = seq_len

#     # Precompute context masking
#     context_length = context_ids.shape[1]
#     context_mask = torch.arange(seq_len, device=device) < context_length

#     for _ in range(max_length):
#         with torch.no_grad():
#             outputs = model(buffer[:, :current_length])
#             logits = outputs.logits[:, -1, :]  # Last token logits

#         # Apply context masking
#         full_context_logits = logits.clone()
#         question_only_logits = logits.masked_fill(context_mask.unsqueeze(-1), 0)

#         # Compute adjusted logits
#         adjusted_logits = (1 + alpha) * full_context_logits - alpha * question_only_logits

#         # Select next token
#         if temperature == 0:
#             next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)
#         else:
#             probs = F.softmax(adjusted_logits / temperature, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#         if next_token.shape[0] != buffer.shape[0]:  
#             next_token = next_token[0].unsqueeze(0)  # Ensure it matches buffer's batch size

#         buffer[:, current_length] = next_token.squeeze(-1)
#         current_length += 1

#         # Stop early if EOS, newline, or "print" is generated
#         if any(t.item() in eos_token | newline_tokens | print_tokens for t in next_token):
#             break

#     # Extract completion
#     completion_ids = buffer[:, seq_len:current_length]
#     output = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
#     #pdb.set_trace()
#     return completion_ids, output

def nucleus_sampling(logits, top_p=0.9):
    """
    Apply nucleus (top-p) sampling: Keep the top tokens that sum to p% probability.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask out tokens beyond top_p threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()  # Shift mask one position
    sorted_indices_to_remove[:, 0] = False  # Always keep the first token

    # Apply mask
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Sample from filtered distribution
    probs = F.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return torch.gather(sorted_indices, -1, next_token)  # Map sampled index back to original indices

def context_aware_sampling_fast(input_ids, context_ids, model, tokenizer, alpha=0.9, max_length=128, temperature=0.0, nucleus_sample=True):
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
            if nucleus_sample:
                adjusted_logits = adjusted_logits / temperature  # Apply temperature scaling
                next_token = nucleus_sampling(adjusted_logits, top_p=top_p)
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
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure correct padding behavior
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True,device_map="auto")

    context = "Write a python function to remove first and last occurrence of a given character from the string. Return only the function, with no other explanation, print statements, or unit tests.\n\n"
    question = "def remove_Occ(s,ch):\n"
    context_input = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
    question_input = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

    input_ids = torch.cat([context_input, question_input], dim=-1)

    model.eval()
    # standard_output = standard_decoding(input_ids, model)
    context_aware_output = context_aware_sampling_fast(
                                            input_ids,
                                            context_input,
                                            model,
                                            tokenizer,
                                            alpha=0.9,
                                            max_length=512,
                                            temperature=1.0,
                                        )


    # print("__" * 50)
    # print("Standard Decoding Output:\n", standard_output)
    print("__" * 50)
    print("Context-Aware Decoding Output:\n", context_aware_output)