import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~16GB @ BF16, https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
        'mistralai/Ministral-8B-Instruct-2410',
        # ~47GB @ BF16, https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
        'mistralai/Mistral-Small-24B-Instruct-2501',
        # ~204GB @ BF16, https://huggingface.co/mistralai/Mistral-Large-Instruct-2411
        'mistralai/Mistral-Large-Instruct-2411',
        # ~25GB @ BF16, https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        'mistralai/Mistral-Nemo-Instruct-2407'
    ]
    model_name = model_names[0]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=True)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, **gen_args)
