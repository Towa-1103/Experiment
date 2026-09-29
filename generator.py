from datetime import datetime
import math
import json
import torch
from sentence_transformers import util

#モデルとJSONファイルを受けとり返答生成するためのクラス
class ResponseGenerator:
    def __init__(self, model, tokenizer, embed_model, memory_file="memories.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.embed_model = embed_model
        self.memory_file = memory_file
        self.memories = self._load_memories()
        
        # 起動時にすべての記憶をベクトル化してメモリに保持（意味軸の検索高速化）
        if self.memories:
            memory_texts = [m.get("text", "") for m in self.memories]
            self.memory_embeddings = self.embed_model.encode(memory_texts, convert_to_tensor=True)
        else:
            self.memory_embeddings = None

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

    # ==========================================
    # 各軸のスコア計算関数
    # ==========================================
    def _calc_recency_score(self, memory):
        """1. Recency（新しさ）: 指数減衰により 0.0〜1.0 に自動で収まる"""
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
        """2. Importance（重要度）: 1〜10 を 0.0〜1.0 へ静的Min-Max正規化"""
        importance = memory.get("importance", 5)
        # 数式: (x - Min) / (Max - Min)
        return float(importance - 1.0) / 9.0

    def _calc_sender_score(self, memory, current_speaker):
        """3. Sender_id（話者の一致）: 一致で 1.0、不一致で 0.0"""
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
        score_threshold: float = 1.2, # 新しい相場に合わせてデフォルトを1.2に仮設定
        scoring_method: str = "4-axis", 
        scoring_weights: dict = None
    ) -> dict:
        
        if scoring_weights is None:
            scoring_weights = {"relevance": 1.0, "recency": 0.2, "importance": 0.1, "sender_id": 0.5}

        used_texts = []
        used_scores = []
        used_past_replies = []

        if self.memories and self.memory_embeddings is not None:
            query_emb = self.embed_model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, self.memory_embeddings)[0]
            
            scored_memories = []
            for i, memory in enumerate(self.memories):
                # Relevanceはそのまま 0.0〜1.0 として扱う
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
                    "past_reply": memory.get("past_reply", ""),
                    "score": final_score
                })
            
            scored_memories = sorted(scored_memories, key=lambda x: x["score"], reverse=True)
            valid_memories = [m for m in scored_memories if m["score"] >= score_threshold][:top_k]
            
            used_texts = [m["text"] for m in valid_memories]
            used_scores = [f"{m['score']:.4f}" for m in valid_memories]
            used_past_replies = [m["past_reply"] for m in valid_memories if m.get("past_reply")]

        # ==========================================
        # 動的ルーティングとプロンプト生成
        # ==========================================
        if len(used_texts) > 0:
            memory_context = "\n".join([f"- {t}" for t in used_texts])
            past_reply_context = "\n".join([f"- {r}" for r in used_past_replies if r])
            
            system_prompt = (
                "あなたは日本人です。必ず自然な日本語のみで返答してください。\n"
                "あなたはユーザーの友人としてLINEで会話をしています。\n"
                "以下の過去の記憶（文脈）と、口調はあなたの過去の返答例を参考にして、相手のメッセージに自然に返信してください。\n\n"
                f"【過去の記憶】\n{memory_context}\n\n"
                f"【あなたの過去の返答例】\n{past_reply_context}\n\n"
                "指示:\n"
                "- LINEらしい短くカジュアルな口調を守ること。\n"
                "- 記憶にないことは適当にでっち上げず、自然に会話を繋ぐこと。"
            )
        else:
            system_prompt = (
                "あなたは日本人です。必ず自然な日本語のみで返答してください。\n"
                "あなたはユーザーの友人としてLINEで会話をしています。\n"
                "現在、相手の話題に関する過去の記憶がありません。\n\n"
                "指示:\n"
                "- LINEらしい短くカジュアルな口調を守ること。\n"
                "- 話題がわからない場合は、「ごめん、それいつの話だっけ？」「ちょっと覚えてないかもです」と自然に聞き返すこと。"
            )

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

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
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