from datetime import datetime
import json
import re
from google import genai

EXTRACT_ALL_PROMPT = """あなたは対話ログから有益な記憶を網羅的かつ高精度に抽出する専門エンジンです。
以下の会話ログを分析し、含まれている【予定・約束・話題・共有されたエピソード】を漏れなくすべて抽出してJSON配列で出力してください。

【会話ログ】
{dialogue}

【厳格な抽出・分割ルール】
1. 複数件の網羅的抽出:
   - 会話内に複数の話題や時間帯のやり取りがある場合は、1つにまとめず【必ず話題ごとに分割して複数件】出力してください。
2. sender_id と 話者:
   - sender_id には【対話相手の名前・表示名】（例: "友人A", "田中" など、ログに登場する相手の名前）を設定してください。「自分」を入れるのは厳禁です。
   - past_reply には【自分（ログ上で「自分」と書かれた発言）】から、口調模倣に使える特徴的な発言（5文字以上）を引用してください。相手の発言を入れるのは厳禁です。
3. 要約（text）の品質:
   - コピペや「〜について話した」だけの曖昧な要約は禁止。「誰が・何を・どうするのか」を客観的かつ具体的に要約してください（40〜80文字）。
4. 除外対象:
   - 「了解」「おつかれ」などの単なる挨拶・相槌、意味のない内輪ノリだけのやり取りは抽出しないでください。

【出力フォーマット例】
[
  {{
    "timestamp": "2026-04-10 12:30:00",
    "sender_id": "友人A",
    "text": "来週金曜日の全員参加の顔合わせについて、集合時間や居酒屋への参加可否を確認した",
    "past_reply": "やったりしましょう！楽しみにしてる",
    "importance": 7
  }},
  {{
    "timestamp": "2026-04-10 21:15:00",
    "sender_id": "友人A",
    "text": "友人Aが研究室の課題の進捗について相談してきたため、互いに手をつけていない状況を共有した",
    "past_reply": "俺も全然終わってないから明日やろ",
    "importance": 5
  }}
]

【出力形式】
Markdown（```json など）や説明文は一切含めず、純粋な `[` で始まり `]` で終わる JSON 配列のみを出力してください。
"""


def extract_memories_with_gemini(
    raw_log, api_key, default_sender="friend", min_importance=3
):
  """Gemini API を使用して会話ログから記憶を高精度に抽出する"""
  client = genai.Client(api_key=api_key)

  prompt = EXTRACT_ALL_PROMPT.format(dialogue=raw_log.strip())

  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    extracted_list = json.loads(response.text)
  except Exception as e:
    print(f"Gemini API 呼び出しエラー: {e}")
    return []

  if not isinstance(extracted_list, list):
    return []

  valid_memories = []
  now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  for item in extracted_list:
    if item.get("importance", 1) >= min_importance:
      if not item.get("timestamp"):
        item["timestamp"] = now_str

      # 相手の名前が入っていない、または誤って「自分」になっている場合の安全策
      sender = item.get("sender_id", "").strip()
      if not sender or sender == "自分":
        item["sender_id"] = default_sender

      valid_memories.append(item)

  return valid_memories