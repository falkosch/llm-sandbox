import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~5.396GB @ BF16, https://huggingface.co/google/recurrentgemma-2b-it
        # mostly english responses
        'google/recurrentgemma-2b-it',
        # ~19.3GB @ BF16, https://huggingface.co/google/recurrentgemma-9b-it
        'google/recurrentgemma-9b-it',
    ]
    model_name = model_names[1]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=False)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, merge_system_prompt_into_first_user_prompt=True, **gen_args)
