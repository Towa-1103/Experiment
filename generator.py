from datetime import datetime
import math
import json
import torch
from sentence_transformers import util

class ResponseGenerator:
    def __init__(self, model, tokenizer, embed_model, memory_file="memories.json", rules_file="person_rules.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.embed_model = embed_model
        
        # 1. エピソード記憶の読み込みとベクトル化（既存の高速化ロジックを維持）
        self.memory_file = memory_file
        self.memories = self._load_memories()
        
        if self.memories:
            memory_texts = [m.get("text", "") for m in self.memories]
            self.memory_embeddings = self.embed_model.encode(memory_texts, convert_to_tensor=True)
        else:
            self.memory_embeddings = None

        # 2. セマンティック記憶（関係性ルール）の読み込みを追加
        self.rules_file = rules_file
        self.person_rules = self._load_rules()

    def _load_memories(self):
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: {self.memory_file} が見つかりません。")
            return []
        except json.JSONDecodeError:
            print(f"エラー: {self.memory_file} のフォーマットが不正です。")
            return []

    def _load_rules(self):
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"警告: {self.rules_file} が見つからないか不正です。デフォルトルールを使用します。")
            return {
                "default": {
                    "relation": "初対面または関係性が薄い相手",
                    "tone_rules": "標準的な丁寧語（〜です、〜ます）で親切に接する。絵文字は控えめにし、英語は絶対に混ぜないこと。"
                }
            }

    # ==========================================
    # 各軸のスコア計算関数（既存ロジックをそのまま活用）
    # ==========================================
    def _calc_recency_score(self, memory):
        time_str = memory.get("timestamp")
        if not time_str:
            return 0.0
        try:
            memory_date = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            delta = now - memory_date
            days_passed = max(0, delta.days)
            return math.exp(-0.0038 * days_passed)
        except ValueError:
            return 0.0

    def _calc_importance_score(self, memory):
        importance = memory.get("importance", 5)
        return float(importance - 1.0) / 9.0

    def _calc_sender_score(self, memory, current_speaker):
        memory_sender = memory.get("sender_id", "")
        return 1.0 if memory_sender == current_speaker else 0.0

    # ==========================================
    # メイン生成処理
    # ==========================================
    def generate_reply(
        self, 
        query: str, 
        current_speaker: str, 
        top_k: int = 2, 
        score_threshold: float = 1.35, # 4軸専用の厳しい閾値にデフォルトを変更
        scoring_method: str = "4-axis", 
        scoring_weights: dict = None
    ) -> dict:
        
        if scoring_weights is None:
            scoring_weights = {"relevance": 1.0, "recency": 0.2, "importance": 0.1, "sender_id": 0.3} # 話者の重みを0.3に調整

        used_texts = []
        used_scores = []

        # --- 1. 記憶の検索と足切り ---
        if self.memories and self.memory_embeddings is not None:
            query_emb = self.embed_model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, self.memory_embeddings)[0]
            
            scored_memories = []
            for i, memory in enumerate(self.memories):
                relevance = cos_scores[i].item()
                final_score = relevance * scoring_weights.get("relevance", 1.0)
                
                if scoring_method in ["3-axis", "4-axis"]:
                    recency = self._calc_recency_score(memory)
                    importance = self._calc_importance_score(memory)
                    final_score += recency * scoring_weights.get("recency", 0.0)
                    final_score += importance * scoring_weights.get("importance", 0.0)
                    
                if scoring_method == "4-axis":
                    sender = self._calc_sender_score(memory, current_speaker)
                    final_score += sender * scoring_weights.get("sender_id", 0.0)

                scored_memories.append({
                    "text": memory.get("text", ""),
                    "score": final_score
                })
            
            scored_memories = sorted(scored_memories, key=lambda x: x["score"], reverse=True)
            valid_memories = [m for m in scored_memories if m["score"] >= score_threshold][:top_k]
            
            used_texts = [m["text"] for m in valid_memories]
            used_scores = [f"{m['score']:.4f}" for m in valid_memories]
            # ※past_replyの抽出処理は完全に削除しました

        # --- 2. ルールの取得 ---
        rule = self.person_rules.get(current_speaker, self.person_rules.get("default"))
        relation = rule.get("relation", "不明")
        tone_rules = rule.get("tone_rules", "自然な日本語で返答してください。")

        # --- 3. システムプロンプトの構築 ---
        system_base = (
            "あなたはLINEのチャットボットとしてユーザーと対話します。\n"
            "以下の【相手との関係性と口調の絶対ルール】を必ず守って返答してください。\n\n"
            f"【相手との関係性と口調の絶対ルール】\n"
            f"相手: {current_speaker}\n"
            f"関係性: {relation}\n"
            f"口調の指示: {tone_rules}\n\n"
            "※上記の口調の指示は、記憶の有無に関わらず常に厳守し、絶対に英語や不自然な言語を混ぜないでください。\n"
        )

        if len(used_texts) > 0:
            memory_context = "\n".join([f"- {t}" for t in used_texts])
            system_prompt = system_base + f"\n【過去の記憶】\n以下の出来事を踏まえて返答してください。\n{memory_context}"
        else:
            system_prompt = system_base + "\n【過去の記憶】\n特になし。今の話題に自然に返答してください。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        text_for_model = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text_for_model], return_tensors="pt").to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]

        # --- 4. 生成（ハルシネーション対策のパラメータへ変更） ---
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.3,         # 創作を抑え、ルールに忠実にするため 0.7 から 0.3 へ変更
                top_p=0.9,
                repetition_penalty=1.1,  # 英語の連続や同じ単語の反復を防ぐため 1.05 から 1.1 へ強化
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=terminators
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        reply = reply.replace("forgettable_id_", "").strip()
        reply = reply.split("\n")[0]

        return {
            "reply": reply,
            "used_memories_count": len(used_texts),
            "used_memories_texts": used_texts,
            "used_memories_scores": used_scores
        }