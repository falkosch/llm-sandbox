import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~1GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
        'Qwen/Qwen2.5-0.5B-Instruct',  # english
        # ~3GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
        'Qwen/Qwen2.5-1.5B-Instruct',
        # ~6GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
        'Qwen/Qwen2.5-3B-Instruct',
        # ~15GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
        'Qwen/Qwen2.5-7B-Instruct',
        # ~30GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
        'Qwen/Qwen2.5-14B-Instruct',
        # ~66GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
        'Qwen/Qwen2.5-32B-Instruct',
        # ~146GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
        'Qwen/Qwen2.5-72B-Instruct',

        # ~1GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
        'Qwen/Qwen2.5-Coder-0.5B-Instruct',
        # ~3GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
        'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        # ~6GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct
        'Qwen/Qwen2.5-Coder-3B-Instruct',
        # ~15GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
        'Qwen/Qwen2.5-Coder-7B-Instruct',
        # ~30GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct
        'Qwen/Qwen2.5-Coder-14B-Instruct',
        # ~66GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        # ~146GB @ BF16, https://huggingface.co/Qwen/Qwen2.5-Coder-72B-Instruct
        'Qwen/Qwen2.5-Coder-72B-Instruct',
    ]
    model_name = model_names[12]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=True)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, **gen_args)
