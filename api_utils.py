# -*- coding: utf-8 -*-
# @Time     :2024/5/21  16:16
# @Author   :fuqihang

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

local_models={
    "Phi-3.5-mini-instruct": "Phi-3.5-mini-instruct",
    "Marco-o1": "Marco-o1",
    "Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "Llama-2-13b-chat-hf",
    "Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
    "phi-2-pytdml": "phi-2-pytdml",
    "Yi-1.5-9B-Chat": "Yi-1.5-9B-Chat",
}

contex_length = {
    "llama-7b": 1536,
    "llama-13b": 1536,
    "Llama-2-7b-chat": 4096,
    "Llama-2-7b-chat-hf": 4096,
    "Llama-2-13b-chat-hf": 4096,
    "Llama2-Chinese-13b-Chat": 4096,
    "Llama-2-70b-chat-hf": 4096,
    "Llama-3-8B-Instruct": 8*1024,
    "Llama-3-70B-Instruct": 8*1024,
    "vicuna-7b-v1.5": 4096,
    "vicuna-13b-v1.5": 4096,
    "falcon-7b-instruct": 2048,
    "falcon-11B": 1536,
    "Mistral-7B-Instruct-v0.2": 1536,
    "Mistral-7B-Instruct-v0.3": 1536,
    "phi-2-pytdml": 2048,
    "Qwen2-7B-Instruct": 131072,
    "gemma-2-9b-it": 4096,
}

class IChatAPI:
    def __init__(self, modelName):

        self.modelName = modelName
        if modelName in local_models.keys():
            self.model = AutoModelForCausalLM.from_pretrained(
                    local_models[modelName],
                    device_map='auto',
                    # use_safetensors=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
            # self.model = load_checkpoint_and_dispatch(self.model, checkpoint=local_models[modelName], device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(
                    local_models[modelName], use_fast=False,
                    add_bos_token=False, add_eos_token=False,
                )
        else:
            raise ValueError("no model named '{}'".format(modelName))

    def __call__(self, *args, **kwargs):
        if self.modelName in local_models.keys():
            return self.grain_probs(*args, **kwargs)
        else:
            raise ValueError("Model not find")


    def convert_chat_to_text(self, messages):
        text = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                text += f"System Instruction: {content}\n\n"
            elif role == "user":
                text += f"User: {content}\n\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n\n"

        return text.strip()


    def preprocess_messages(self, messages):
        sys_messages = messages[0]['content']
        res_messages = []
        for message in messages:
            if message["role"] == "user":
                res_messages.append({'role': message['role'], 'content': sys_messages + '\n' + message['content']})
            if message["role"] == "assistant":
                res_messages.append(message)
        return res_messages


    def grain_probs(self, messages, candidate):

        # messages_text = self.convert_chat_to_text(messages)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            # add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        # input_ids = input_ids[..., -1536:]
        input_ids = input_ids[..., -1024*8:]
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
            ).logits[:, -1].view(-1)

        option_indices = [self.tokenizer(f': {e}').input_ids[-1] for e in candidate] + \
                         [self.tokenizer(f':{e}').input_ids[-1] for e in candidate]

        probs = F.softmax(
            logits[..., option_indices], dim=-1
        ).detach().cpu().to(torch.float32).numpy()
        probs = probs.reshape(2, len(candidate)).sum(axis=0)
        return list(zip(candidate, [float(p) for p in probs]))


    def generate(self, messages, max_tokens, do_sample=True, temperature=0.1, top_p=0.1):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            # top_p=top_p
        )
        response = outputs[0][input_ids.shape[-1]:]

        return self.tokenizer.decode(response, skip_special_token=True)
