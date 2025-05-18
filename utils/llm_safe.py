import time
import tiktoken

def truncate_text_to_token_limit(text, max_tokens, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])

def safe_chat_completion(
    client,
    model,
    messages,
    max_tokens=1024,
    temperature=0.7,
    sleep_seconds=5,                # To avoid hitting TPM rate limits
    prompt_truncate_tokens=6000     # Truncate long inputs to avoid model limit
):
    """
    Wrapper around OpenAI ChatCompletion with sleep + input truncation.
    """
    if prompt_truncate_tokens is not None:
        for m in messages:
            if m["role"] == "user":
                m["content"] = truncate_text_to_token_limit(m["content"], prompt_truncate_tokens, model)

    time.sleep(sleep_seconds)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

