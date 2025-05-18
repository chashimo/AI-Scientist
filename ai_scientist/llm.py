import json
import os
import re
import anthropic
import backoff
import openai
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from utils.llm_safe import safe_chat_completion

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "llama3.1-405b",
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
]

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(msg, client, model, system_message, print_debug=False, msg_history=None, temperature=0.75, n_responses=1):
    if msg_history is None:
        msg_history = []
    if 'gpt' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        messages = [{"role": "system", "content": system_message}, *new_msg_history]
        response = safe_chat_completion(client=client, model=model, messages=messages, max_tokens=MAX_NUM_TOKENS, temperature=temperature)
        content = [r.message.content for r in response.choices]
        new_msg_history = [new_msg_history + [{"role": "assistant", "content": c}] for c in content]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(msg, client, model, system_message, print_debug=False, msg_history=None, temperature=temperature)
            content.append(c)
            new_msg_history.append(hist)
    if print_debug:
        print("\n" + "*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21 + "\n")
    return content, new_msg_history

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(msg, client, model, system_message, print_debug=False, msg_history=None, temperature=0.75):
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    messages = [{"role": "system", "content": system_message}, *new_msg_history]
    response = safe_chat_completion(client=client, model=model, messages=messages, max_tokens=MAX_NUM_TOKENS, temperature=temperature)
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    if print_debug:
        print("\n" + "*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21 + "\n")
    return content, new_msg_history

def extract_json_between_markers(llm_output):
    matches = re.findall(r"```json(.*?)```", llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"\{.*?\}", llm_output, re.DOTALL)
    for json_string in matches:
        try:
            return json.loads(json_string.strip())
        except json.JSONDecodeError:
            try:
                clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(clean)
            except json.JSONDecodeError:
                continue
    return None

def create_client(model):
    if model.startswith("claude-"):
        return anthropic.Anthropic(), model
    if model.startswith("bedrock") and "claude" in model:
        return anthropic.AnthropicBedrock(), model.split("/")[-1]
    if model.startswith("vertex_ai") and "claude" in model:
        return anthropic.AnthropicVertex(), model.split("/")[-1]
    if 'gpt' in model or "o1" in model or "o3" in model:
        return openai.OpenAI(), model
    if model.startswith("deepseek"):
        return openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"), model
    if model == "llama3.1-405b":
        return openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"), "meta-llama/llama-3.1-405b-instruct"
    if "gemini" in model:
        return openai.OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/"), model
    raise ValueError(f"Model {model} not supported.")

