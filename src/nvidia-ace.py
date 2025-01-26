import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~4GB @ BF16, https://huggingface.co/nvidia/AceInstruct-1.5B
        'nvidia/AceInstruct-1.5B',
        # ~16GB @ BF16, https://huggingface.co/nvidia/AceInstruct-7B
        'nvidia/AceInstruct-7B',
        # ~146GB @ BF16, https://huggingface.co/nvidia/AceInstruct-72B
        'nvidia/AceInstruct-72B',
    ]
    model_name = model_names[1]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=True)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, **gen_args)
