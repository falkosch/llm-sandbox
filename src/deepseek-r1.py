import transformers as tf

from __init__ import DEFAULT_MODEL_KWARGS, DEFAULT_GEN_ARGS, read_json, eval_prompt

if __name__ == "__main__":
    model_names = [
        # ~3.5GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        # ~15GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        # ~30GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        # ~66GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        # ~16GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        # ~141GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        # ~689GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1
        'deepseek-ai/DeepSeek-R1',
        # ~689GB @ BF16, https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero
        'deepseek-ai/DeepSeek-R1-Zero'
    ]
    model_name = model_names[4]
    model_kwargs = DEFAULT_MODEL_KWARGS | {}
    gen_args = DEFAULT_GEN_ARGS | {}

    model = tf.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    pipeline = tf.pipeline('text-generation', model, model.config, tokenizer, use_cache=True)

    request = read_json('data/describe_company_bakery.json5')
    eval_prompt(model, tokenizer, pipeline, request, **gen_args)
