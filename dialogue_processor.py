from datetime import datetime
import json
import re
import torch

EXTRACT_ALL_PROMPT = """あなたは対話ログから有益な記憶を網羅的に抽出するエンジンです。
以下の会話ログを分析し、含まれている【予定・約束・話題・共有された情報】を漏れなくすべて抽出してJSON配列で出力してください。

【会話ログ】
{dialogue}

【分割・抽出ルール】
1. 複数抽出の徹底:
   - 1つの会話ログの中に複数の話題や異なる時間帯のやり取りがある場合、絶対に1つにまとめず、【話題ごとに別々のオブジェクト】として複数件抽出してください。
2. sender_id と 話者:
   - 相手はすべて "friend" に統一してください。
   - past_reply は【自分（ログ上で「自分」の発言）】から、口調がわかる意味のある発言（5文字以上）を引用してください。相手の発言を入れるのは厳禁です。
3. 要約（text）の品質:
   - 発言のコピペや「〜について話した」だけの曖昧な文は禁止。「誰が・何を・どうするのか」を客観的に要約してください（30〜60文字）。
4. 除外対象:
   - 「了解」「おつかれ」などの単なる挨拶・相槌、意味のない内輪ノリだけのやり取りは抽出しないでください。

【出力フォーマット例（必ずこのように複数件に分けて出力してください）】
[
  {{
    "timestamp": "2026-04-10 12:30:00",
    "sender_id": "friend",
    "text": "来週金曜日の顔合わせの集合時間と居酒屋への参加について確認した",
    "past_reply": "やったりしましょう！楽しみにしてる",
    "importance": 7
  }},
  {{
    "timestamp": "2026-04-10 21:15:00",
    "sender_id": "friend",
    "text": "友人が研究室の課題で詰まっていると相談を受け、進捗状況を共有した",
    "past_reply": "俺も全然終わってないから明日やろ",
    "importance": 5
  }}
]

【出力形式】
解説やマークダウンは一切含めず、純粋な `[` で始まり `]` で終わる JSON 配列のみを出力してください。
"""

def extract_memories_from_log(full_text, tokenizer, model, min_importance=3):
  """会話ログ全体から重要エピソードを要約抽出する関数"""
  prompt = EXTRACT_ALL_PROMPT.format(dialogue=full_text.strip())
  messages = [
      {
          "role": "system",
          "content": (
              "あなたは対話ログからエピソードを要約抽出し、JSON配列のみを出力するエンジンです。"
          ),
      },
      {"role": "user", "content": prompt},
  ]

  text_input = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

  with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

  response = tokenizer.decode(
      outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
  )

  match = re.search(r"\[.*\]", response, re.DOTALL)
  extracted_list = None

  if match:
    try:
      extracted_list = json.loads(match.group(0))
    except json.JSONDecodeError:
      pass

  # 末尾途切れ救済
  if extracted_list is None:
    last_bracket = response.rfind("}")
    first_bracket = response.find("[")
    if first_bracket != -1 and last_bracket != -1:
      salvaged_str = response[first_bracket : last_bracket + 1] + "]"
      try:
        extracted_list = json.loads(salvaged_str)
      except json.JSONDecodeError:
        return []
    else:
      return []

  if not isinstance(extracted_list, list):
    return []

  valid_memories = []
  now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  for item in extracted_list:
    score = item.get("importance", 1)
    if score >= min_importance:
      if not item.get("timestamp"):
        item["timestamp"] = now_str
      valid_memories.append(item)

  return valid_memories