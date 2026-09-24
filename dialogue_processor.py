import json
import re
import torch

ANALYSIS_PROMPT = """以下のLINEの会話ログを分析し、指定のJSONフォーマットのみを出力してください。前置きや解説、マークダウンの装飾記法は一切含めないでください。

【会話ログ】
{dialogue}

【判定ルール】
1. timestamp: 会話ログから日付・時刻を読み取り "YYYY-MM-DD HH:MM:SS" 形式で出力（年が不明なら2026年、時刻のみなら日付を補完、完全に不明なら空文字）
2. sender_id: 会話相手を推測して分類 ("friend", "parent", "professor", "other" のいずれか)
3. text: この会話で決まったことや話した内容の客観的要約（30〜60文字程度）
4. past_reply: 会話内での「自分」の発言（口調模倣用、なければ空文字）
5. importance: 記憶の重要度（1〜10の整数）
   - 1〜2: 「了解」「それな」「スタンプ」など中身のない相槌・挨拶
   - 3〜5: 日常の雑談、ちょっとした予定合わせ、近況報告
   - 6〜8: 試験・進路・研究の相談、重要な約束
   - 9〜10: 人生の重大事、深刻な相談

【出力フォーマット】
{{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "sender_id": "...",
  "text": "...",
  "past_reply": "...",
  "importance": 整数
}}
"""


def process_dialogue_chunk(chunk_text, tokenizer, model, threshold=3):
  """会話ブロックを分析し、重要度が閾値以上ならJSONデータを返し、未満ならNoneを返す関数"""
  prompt = ANALYSIS_PROMPT.format(dialogue=chunk_text.strip())
  messages = [
      {
          "role": "system",
          "content": (
              "あなたは対話テキストを分析して正確なJSONのみを出力するデータ処理エンジンです。"
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
        max_new_tokens=256,
        do_sample=False,
    )

  response = tokenizer.decode(
      outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
  )

  match = re.search(r"\{.*\}", response, re.DOTALL)
  if not match:
    return None

  try:
    result = json.loads(match.group(0))
  except json.JSONDecodeError:
    return None

  if not result.get("timestamp"):
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  score = result.get("importance", 1)

  if score < threshold:
    return None

  return result