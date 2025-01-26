import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, eval_prompt, read_json

if __name__ == "__main__":
    model_names = [
        # ~208GB @ BF16, https://huggingface.co/CohereForAI/c4ai-command-r-plus
        'CohereForAI/c4ai-command-r-plus',
    ]
    model_name = model_names[0]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=True)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, **gen_args)
