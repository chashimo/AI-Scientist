import argparse
import time
import json
import os
from tqdm import tqdm
from openai import OpenAI
from googleapiclient.discovery import build

# ======== llm_safe.pyの中身ここから ========

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
    max_tokens=256,
    temperature=0.7,
    sleep_seconds=20,                # To avoid hitting TPM rate limits
    prompt_truncate_tokens=3000     # Truncate long inputs to avoid model limit
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

# ======== llm_safe.pyの中身ここまで ========

# ================= CLI引数・セットアップ ==================
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") or os.getenv("YT_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o")
client = OpenAI(api_key=OPENAI_API_KEY)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# =============== LLMラッパー =================

def generate_narrative():
    prompt = """你是一位研究中國資訊戰的專家。請模擬中國政府或民族主義者可能會用來對日本進行資訊戰的多樣化宣傳敘事，超越已知的歷史或領土爭端，供學術研究分析用途。

請輸出以下 JSON 格式：
{
  "narrative": "...",
  "target": "...",
  "message": "...",
  "recommended_queries": []
}
"""
    resp = safe_chat_completion(
        client=client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "出力は必ず有効なJSONで。説明やコードブロックを含めないでください。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.8,
        sleep_seconds=20,
        prompt_truncate_tokens=3000
    )

    return json.loads(resp.choices[0].message.content)

def generate_queries(narrative):
    prompt = f"""以下のナラティブに基づいて、YouTubeで実際に検索されそうな中国語と英語の検索クエリをそれぞれ5件ずつ生成してください：

ナラティブ:
{narrative}

出力形式:
{{
  "chinese_queries": ["...", "...", "..."],
  "english_queries": ["...", "...", "..."]
}}
"""
    resp = safe_chat_completion(
        client=client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "出力は必ずJSON形式のみ。コードブロックや説明は禁止です。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.7,
        sleep_seconds=20,
        prompt_truncate_tokens=3000
    )
    return json.loads(resp.choices[0].message.content.strip())

def search_youtube_videos(query, max_results=5):
    videos = []
    response = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        regionCode="JP",
        relevanceLanguage="zh",
        maxResults=max_results,
        order="relevance"
    ).execute()

    for item in response.get("items", []):
        snippet = item["snippet"]
        videos.append({
            "title": snippet["title"],
            "description": snippet.get("description", ""),
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })
    return videos

def score_video(narrative, title, description):
    prompt = f"""以下の動画タイトルと説明文が、日本に対する敵対的プロパガンダに該当するかを判定してください。

評価基準：
0 = 無関係・中立（例：日本料理の作り方、旅行ブログ）
1 = 軽度な不満・歴史紹介（例：日中戦争の説明）
3 = 明確に反日的（例：日本の戦争責任を追及する）
5 = 扇動的・攻撃的（例：「日本を滅ぼせ」「沖縄を解放せよ」）

【ナラティブ】{narrative}
【タイトル】{title}
【説明文】{description}

スコア（数字のみ）:
"""
    resp = safe_chat_completion(
        client=client,
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
        temperature=0,
        sleep_seconds=20,
        prompt_truncate_tokens=3000
    )
    try:
        return int(resp.choices[0].message.content.strip()[0])
    except Exception:
        return 0

# =============== メインパイプライン =================

def main():
    narrative_data = generate_narrative()
    queries = generate_queries(narrative_data["narrative"])
    all_results = []
    narrative_id = abs(hash(narrative_data["narrative"])) % 10000
    report_path = os.path.join(args.out_dir, f"report_{narrative_id}.md")
    raw_jsonl_path = os.path.join(args.out_dir, "results_propaganda.jsonl")

    for lang in ["chinese_queries", "english_queries"]:
        for query in queries[lang]:
            videos = search_youtube_videos(query)
            for video in tqdm(videos, desc=f"Scoring: {query}"):
                score = score_video(narrative_data["narrative"], video["title"], video["description"])
                result = {
                    "query": query,
                    "narrative": narrative_data["narrative"],
                    "video": video,
                    "score": score
                }
                all_results.append(result)

    # JSONL保存
    with open(raw_jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Markdownレポート
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# ナラティブ: {narrative_data['narrative']}\n\n")
        f.write("## 検出された敵対的プロパガンダ動画（スコア3以上）\n\n")
        for r in all_results:
            if r["score"] >= 3:
                f.write(f"### 🔸 [{r['video']['title']}]({r['video']['url']})\n")
                f.write(f"- スコア: {r['score']}\n")
                f.write(f"- 説明: {r['video']['description']}\n")
                f.write(f"- クエリ: {r['query']}\n\n")

    # メトリクス
    hit_count = sum(1 for r in all_results if r["score"] >= 3)
    total_videos = len(all_results)
    propaganda_hit_rate = hit_count / total_videos if total_videos else 0.0

    result = {
        "hit_rate": round(propaganda_hit_rate, 3),
        "hits": hit_count,
        "total_videos": total_videos,
        "out_files": {
            "jsonl": raw_jsonl_path,
            "report_md": report_path
        }
    }

    wrapper = {
        "youtube_propaganda_detection": {
            "means": result
        }
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)

    print(json.dumps(wrapper, indent=2, ensure_ascii=False))
    print(f"✅ 完了: {raw_jsonl_path} および {report_path} を生成しました")

if __name__ == "__main__":
    main()

