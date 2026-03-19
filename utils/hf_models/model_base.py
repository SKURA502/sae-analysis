from typing import List, Dict
import torch

from abc import ABC, abstractmethod


class ModelBase(ABC):

    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name

        self.model: AutoModelForCausalLM = self._load_model(model_path)

        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
    
    @abstractmethod
    def _load_model(self, model_path, dtype):
        pass

    @abstractmethod
    def _load_tokenizer(self, model_path):
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_embed_modules(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_model_norm_modules(self):
        pass

    @abstractmethod
    def _get_model_lm_head(self):
        pass

    @torch.no_grad()
    def generate_completions(self, instructions, outputs=None, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64, generation_config=None):

        if generation_config is None:
            generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
            generation_config.pad_token_id = self.tokenizer.pad_token_id
            generation_config.padding_side = self.tokenizer.padding_side

        completions = []

        for i in tqdm(range(0, len(instructions), batch_size)):
            if outputs is not None:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size], outputs=outputs[i:i+batch_size])
            else:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):

                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append(self.tokenizer.decode(generation, skip_special_tokens=True).strip())

        return completions

    def get_logits(self, instructions, completions=None, fwd_pre_hooks=[], fwd_hooks=[], save_positions=[-1], batch_size=8):
        logits_list = []
        for i in tqdm(range(0, len(instructions), batch_size)):
            if completions is not None:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size], outputs=completions[i:i+batch_size])
            else:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
            
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                # Move inputs to the model's device
                input_ids = tokenized_instructions.input_ids.to(self.model.device)
                attention_mask = tokenized_instructions.attention_mask.to(self.model.device)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract logits for the last token
                batch_logits = outputs.logits[:, save_positions, :]
                logits_list.append(batch_logits)
        
        # Concatenate all batches
        all_logits = torch.cat(logits_list, dim=0)
        return all_logits

    @torch.no_grad()
    def generate_with_positional_hooks(self, instructions, steering_substrings: List[List[str]], last_substring_pos, outputs=None, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64, generation_config=None):
        
        if generation_config is None:
            generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
            generation_config.pad_token_id = self.tokenizer.pad_token_id
            generation_config.padding_side = self.tokenizer.padding_side

        completions = []

        for i in tqdm(range(0, len(instructions), batch_size)):
            if outputs is not None:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size], outputs=outputs[i:i+batch_size])
            else:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

            input_ids = tokenized_instructions.input_ids.to('cuda')
            attention_mask = tokenized_instructions.attention_mask.to('cuda')

            n_prompts, seq_len = input_ids.shape
            steering_positions = [[] for _ in range(n_prompts)]

            for batch_idx, active_substrings in enumerate(steering_substrings[i:i+batch_size]):
                for active_substring in active_substrings:
                    pos_slice: slice = find_string_in_tokens(active_substring, input_ids[batch_idx], self.tokenizer)
                    if last_substring_pos:
                        steering_positions[batch_idx].extend([pos_slice.stop-1])
                    else:
                        steering_positions[batch_idx].extend(list(range(pos_slice.start, pos_slice.stop)))

                # remove duplicate indexes
                steering_positions[batch_idx] = list(set(steering_positions[batch_idx]))

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks, steering_positions=steering_positions, seq_len=seq_len):
                generation_toks = self.model.generate(
                    input_ids=input_ids.to(self.model.device),
                    attention_mask=attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append(self.tokenizer.decode(generation, skip_special_tokens=True).strip())

        return completions

    def get_logits_with_positional_hooks(self, instructions, steering_substrings: List[List[str]], last_substring_pos, completions=None, fwd_pre_hooks=[], fwd_hooks=[], save_positions=[-1], batch_size=8):
        logits_list = []
        for i in tqdm(range(0, len(instructions), batch_size)):
            if completions is not None:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size], outputs=completions[i:i+batch_size])
            else:
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
            
            # Move inputs to the model's device
            input_ids = tokenized_instructions.input_ids.to(self.model.device)
            attention_mask = tokenized_instructions.attention_mask.to(self.model.device)

            seq_len = input_ids.size(-1)
            steering_positions = [[] for _ in range(seq_len)]

            for j, active_substrings in enumerate(steering_substrings[i:i+batch_size]):
                for active_substring in active_substrings:
                    pos_slice: slice = find_string_in_tokens(active_substring, input_ids[j], self.tokenizer)
                    if last_substring_pos:
                        # print('steering_positions', steering_positions)
                        # print('active_substring', active_substring)
                        steering_positions[j].extend([pos_slice.stop-1])
                    else:
                        steering_positions[j].extend(list(range(pos_slice.start, pos_slice.stop)))
                
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract logits for the last token
                batch_logits = outputs.logits[:, save_positions, :]
                logits_list.append(batch_logits)
        
        # Concatenate all batches
        all_logits = torch.cat(logits_list, dim=0)
        return all_logits