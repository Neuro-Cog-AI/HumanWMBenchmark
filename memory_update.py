import argparse
import sys
import torch

from transformers import AutoTokenizer

from cbrrnn import CbrRnn
from cbrrnnm import CbrRnnM


def calculate_update(name, architecture, corpus):
    if architecture == "cbrrnn":
        model = CbrRnn.from_pretrained(name)
    elif architecture == "cbrrnnm":
        model = CbrRnnM.from_pretrained(name)
    else:
        raise NotImplementedError(f"Architecture {architecture} not supported")
    tokenizer = AutoTokenizer.from_pretrained(name)
    seq_len = model.max_seq_len
    stride = seq_len // 2  # The window slides by half the sequence length
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_article(text_chunk):
        if not text_chunk.strip():
            return

        encoding = tokenizer(text_chunk, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Input sequence starts with BOS, target sequence is the actual tokens
        inputs = [bos_idx] + token_ids + [eos_idx]
        
        # Slide the window across the article
        for start_idx in range(0, len(inputs), stride):
            end_idx = min(start_idx + seq_len, len(inputs))
            win_inputs = inputs[start_idx:end_idx]
            input_tensor = torch.tensor([win_inputs], device=device)
            _, _, mem_updates = model(input_tensor) 

            # For all subsequent windows, only score the second half
            score_start_in_window = 1 if start_idx == 0 else stride

            for j in range(score_start_in_window, len(win_inputs)):
                global_input_idx = start_idx + j
                token_idx = win_inputs[j]
                if token_idx == eos_idx:
                    continue
                token_idx = global_input_idx - 1
                token_str = tokens[token_idx].replace("Ġ", "")
                update = mem_updates[j].item()
                print(f"{token_str} {update:.6f}")
               
            if end_idx == len(inputs):
                break

    print("word update")
    
    with torch.no_grad():
        with open(corpus, 'r') as f:
            article_text = ""
            
            for line in f:
                stripped_line = line.strip()
                if not stripped_line: 
                    continue
                
                if stripped_line == "!ARTICLE":
                    process_article(article_text.strip())
                    print("--- Article Boundary ---", file=sys.stderr)
                    article_text = ""
                    continue

                # Accumulate raw text, preserving necessary spacing for BPE
                article_text += stripped_line + " "
            
            # Process the final article if the file doesn't end with !ARTICLE
            process_article(article_text.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("architecture", help="model architecture", choices=["cbrrnn", "cbrrnnm"])
    parser.add_argument("corpus", help="corpus on which to calculate surprisal")
    args = parser.parse_args()
    calculate_update(args.model, args.architecture, args.corpus)
