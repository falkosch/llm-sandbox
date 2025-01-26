import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~103GB @ BF16, https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct
        # quality = 3/5 | time = 324.091s | cuda max mem = 31773.3MB
        'nvidia/Llama-3_1-Nemotron-51B-Instruct',
        # ~141GB @ BF16, https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
        'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    ]
    model_name = model_names[0]
    model_kwargs = DEFAULT_MODEL_KWARGS | {
        'trust_remote_code': True,  # :(
    }
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=True)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, **gen_args)
