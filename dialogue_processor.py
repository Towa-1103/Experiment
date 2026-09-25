from datetime import datetime
import json
import re
import torch

EXTRACT_ALL_PROMPT = """以下のLINEの会話ログ全体を読み、記録に残すべき【重要なエピソード・予定の約束・決定事項】を要約して抽出してください。

【会話ログ】
{dialogue}

【抽出ルール】
1. 発言を1行ずつ細切れに抜き出すのではなく、一連の会話で「何が決まったか・何が起きたか」を1つのエピソードとして要約してください。
2. 単なる挨拶、相槌、雑談（「なんだろ」「やったりしましょう」等の単文）は除外してください。
3. 抽出する各要素のキー：
   - timestamp: その話題が話された日時 ("YYYY-MM-DD HH:MM:SS" 形式。時刻のみなら会話内の日付と合体させる)
   - sender_id: 会話相手の分類 ("friend", "parent", "professor", "other" のいずれか)
   - text: 決定事項や出来事の客観的な要約（30〜60文字程度）
   - past_reply: その話題に対する「自分」の発言（口調模倣用、なければ空文字）
   - importance: 重要度（3〜10の整数）
4. 出力は必ず以下の「JSON配列形式」のみで行ってください。

【出力フォーマット】
[
  {{
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "sender_id": "...",
    "text": "...",
    "past_reply": "...",
    "importance": 整数
  }}
]
"""


def extract_memories_from_log(full_text, tokenizer, model, min_importance=3):
  """会話ログ全体から重要エピソードを要約抽出する関数"""
  prompt = EXTRACT_ALL_PROMPT.format(dialogue=full_text.strip())
  messages = [
      {
          "role": "system",
          "content": (
              "あなたは対話ログから有益な記憶を網羅的に抽出し、正確なJSON配列のみを出力するデータ処理エンジンです。"
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
        max_new_tokens=1024,  # トークン数を増やして途切れを防止
        do_sample=False,
    )

  response = tokenizer.decode(
      outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
  )

  # JSON配列 [ ... ] の抽出を試みる
  match = re.search(r"\[.*\]", response, re.DOTALL)
  extracted_list = None

  if match:
    try:
      extracted_list = json.loads(match.group(0))
    except json.JSONDecodeError:
      pass

  # もし末尾が途切れてパース失敗した場合、最後の完全なオブジェクトまでを救済
  if extracted_list is None:
    # 完全に閉じた最後の } までを取得して ] で閉じる
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