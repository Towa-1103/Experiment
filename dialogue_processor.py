from datetime import datetime
import json
import re
import torch

EXTRACT_ALL_PROMPT = """以下のLINE会話ログから、後から参照すべき【話題・予定・出来事】のまとまりを抽出してJSON配列で出力してください。

【会話ログ】
{dialogue}

【抽出の定義】
- text: 会話全体から「何が決まったか・何について話していたか」を客観的にまとめた要約（相手の発言のコピペは禁止）
- past_reply: その話題に対する「自分」の発言（口調模倣に使える特徴的な返信を1つ選ぶ）
- timestamp: その話題が話された日時 (YYYY-MM-DD HH:MM:SS)
- sender_id: 相手の分類 ("friend", "parent", "professor", "other")
- importance: 重要度（3〜10）

【出力フォーマット例】
ログ例:
2026.04.10 金曜日
18:00 友人: 来週の金曜、ご飯行かない？
18:02 自分: 行こ行こ！焼肉がいいな
18:05 友人: おっけー予約しとくわ

出力例:
[
  {{
    "timestamp": "2026-04-10 18:00:00",
    "sender_id": "friend",
    "text": "来週金曜日に友人と焼肉に行く約束をした",
    "past_reply": "行こ行こ！焼肉がいいな",
    "importance": 5
  }}
]

【出力上の厳格なルール】
1. 相槌や中身のない雑談だけの話題は無視してください。
2. 出力はマークダウン記法や解説を含めず、JSONの配列 `[...]` のみを出力してください。
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