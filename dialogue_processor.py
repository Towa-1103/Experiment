from datetime import datetime
import json
import re
import torch

EXTRACT_ALL_PROMPT = """以下のLINEの会話ログ全体を分析し、記憶としてデータベースに記録すべき重要なエピソード・約束・決定事項をすべて抽出してください。

【会話ログ】
{dialogue}

【抽出・判定ルール】
1. 中身のない相槌（「それな」「了解」「スタンプ」等）や単なる挨拶・薄い雑談は【完全に無視】してください。
2. 記憶すべき事象ごとに、以下のキーを持つ辞書を作成してください：
   - timestamp: 会話から読み取れる日時 ("YYYY-MM-DD HH:MM:SS" 形式。時刻のみなら会話内の日付と組み合わせ、不明なら空文字)
   - sender_id: 相手の属性 ("friend", "parent", "professor", "other" のいずれか)
   - text: 客観的な要約（30〜60文字程度）
   - past_reply: その時の「自分」の発言（口調模倣用、なければ空文字）
   - importance: 重要度（3〜10の整数。3未満の取るに足らない会話は含めないこと）
3. 抽出結果を「JSONの配列（リスト）」形式のみで出力してください。前置きや解説、マークダウン記法は含めないでください。記憶すべき内容が1つもない場合は空の配列 `[]` を出力してください。

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
  """会話ログ全体から重要なエピソードのみを一括抽出し、辞書のリストとして返す関数"""
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
        max_new_tokens=512,  # 複数件抽出できるようにトークン数を広めに確保
        do_sample=False,
    )

  response = tokenizer.decode(
      outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
  )

  # [ ... ] の配列部分を正規表現で抽出
  match = re.search(r"\[.*\]", response, re.DOTALL)
  if not match:
    return []

  try:
    extracted_list = json.loads(match.group(0))
  except json.JSONDecodeError:
    return []

  if not isinstance(extracted_list, list):
    return []

  valid_memories = []
  now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  for item in extracted_list:
    score = item.get("importance", 1)
    if score >= min_importance:
      # timestampが欠落している場合の安全策
      if not item.get("timestamp"):
        item["timestamp"] = now_str
      valid_memories.append(item)

  return valid_memories