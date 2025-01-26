import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~5GB @ FP16, https://huggingface.co/google/gemma-2-2b-it
        'google/gemma-2-2b-it',
        # ~17GB @ FP16, https://huggingface.co/google/gemma-2-9b-it
        'google/gemma-2-9b-it',
        # ~55GB @ FP16, https://huggingface.co/google/gemma-2-27b-it
        'google/gemma-2-27b-it',
    ]
    model_name = model_names[2]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, merge_system_prompt_into_first_user_prompt=True, **gen_args)
