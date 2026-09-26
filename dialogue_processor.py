from datetime import datetime
import json
import re
import torch

EXTRACT_ALL_PROMPT = """以下のLINE会話ログを分析し、後から参照すべき【話題・予定・エピソード】を漏れなくすべて抽出してJSON配列で出力してください。

【会話ログ】
{dialogue}

【抽出・分割のルール】
1. 会話の中に「複数の異なる話題や予定」がある場合は、1つにまとめず、話題ごとにオブジェクトを分けて【必ず複数件】出力してください。
2. 雑談・相槌だけの行は無視し、中身のあるエピソード（決定事項、約束、趣味や人に関する情報共有など）を対象にしてください。
3. 各エピソードの要約（text）は、発言のコピペではなく前後の文脈を含めて客観的にまとめてください（40〜80文字）。
4. past_reply には、その話題において「自分」が発言した象徴的な返信を1つ引用してください（口調再現用）。

【出力フォーマット例（複数件出力する見本）】
ログ例:
2026.04.10 金曜日
15:00 友人: 来週金曜の顔合わせ、全員参加らしいよ
15:02 自分: やったりましょう。何すんだろ
15:05 友人: 自己紹介とかかな。居酒屋行くなら帰るわ笑
20:00 友人: そういえば例の研究室の課題終わった？
20:05 自分: まだ全然やってないわヤバい

出力例:
[
  {{
    "timestamp": "2026-04-10 15:00:00",
    "sender_id": "friend",
    "text": "来週金曜に全員参加の顔合わせが予定されており、内容や居酒屋の参加について話した",
    "past_reply": "やったりしましょう",
    "importance": 7
  }},
  {{
    "timestamp": "2026-04-10 20:00:00",
    "sender_id": "friend",
    "text": "研究室の課題の進捗について聞かれ、まだ手をつけていない状況を共有した",
    "past_reply": "まだ全然やってないわヤバい",
    "importance": 5
  }}
]

【出力ルール】
- マークダウンや説明文は一切出力せず、必ず `[` で始まり `]` で終わる JSON 配列のみを出力してください。
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