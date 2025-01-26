import logging
import os
import time

import dotenv
import huggingface_hub
import torch
import transformers as tf

dotenv.load_dotenv()
logging.basicConfig(level=logging.DEBUG)
huggingface_hub.login(token=os.getenv("HF_TOKEN"))

DEFAULT_MODEL_DTYPE = torch.float16
DEFAULT_MODEL_KWARGS = {
    'torch_dtype': DEFAULT_MODEL_DTYPE,
    'device_map': 'cuda',
    'quantization_config': tf.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=DEFAULT_MODEL_DTYPE,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_quant_storage=DEFAULT_MODEL_DTYPE,
    ),
}

DEFAULT_GEN_ARGS = {
    'max_new_tokens': 4096,
    'num_return_sequences': 1,
    'do_sample': False,
    'repetition_penalty': 1.0,
}

import pathlib as pl
import typing as tp

import json5
import torch

Prompts = list[dict[str, str]]


def keep_roles(sample: Prompts, roles=None) -> Prompts:
    roles = roles or ['user', 'assistant']
    return [prompt for prompt in sample if prompt['role'] in roles]


def read_json(relative_path: str) -> any:
    with pl.Path.cwd().joinpath(relative_path).open(encoding='utf-8') as file:
        return json5.load(file)


def compile_model(model: tp.Callable) -> tp.Callable:
    return torch.compile(model, mode='max-autotune', fullgraph=True)


def eval_prompt(model: tf.PreTrainedModel, tokenizer: tf.PreTrainedTokenizerBase, pipeline: tf.Pipeline,
                request: Prompts, merge_system_prompt_into_first_user_prompt=False, **kwargs) -> Prompts:
    print(model.__class__, tokenizer.__class__)

    if merge_system_prompt_into_first_user_prompt:
        system_messages = keep_roles(request, roles=['system'])
        if system_messages:
            other_messages = keep_roles(request)
            first_user_message_index = next(
                (i for i, message in enumerate(other_messages) if message['role'] == 'user'), None)
            if first_user_message_index is not None:
                system_prompt = '\n'.join(message['content'] for message in system_messages)
                first_user_message = other_messages[first_user_message_index]['content']
                other_messages[first_user_message_index]['content'] = f'{system_prompt}\n\n{first_user_message}'
                request = other_messages

    logging.info('REQUEST %s', request)

    start_time = time.time()
    response = pipeline(request, **kwargs)

    logging.info('RESPONSE %s\ntime = %gs | cuda max mem = %gMB', response,
                 time.time() - start_time, torch.cuda.max_memory_allocated() / 1024 / 1024)
    return response
