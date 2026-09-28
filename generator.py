import json
import torch
from sentence_transformers import util

class ResponseGenerator:
    def __init__(self, model, tokenizer, embed_model, memory_file="memories.json"):
        """
        Colabの別セルでロードしたモデルを受け取る（依存性の注入）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embed_model = embed_model
        self.memory_file = memory_file
        self.memories = self._load_memories()
        
        # 起動時にすべての記憶をベクトル化してメモリに保持（検索の高速化）
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
            print(f"警告: {self.memory_file} が見つかりません。記憶ゼロで開始します。")
            return []
        except json.JSONDecodeError:
            print(f"エラー: {self.memory_file} のフォーマットが不正です。")
            return []

    def generate_reply(self, query: str, current_speaker: str, top_k: int = 2, score_threshold: float = 0.0) -> dict:
        """
        スコア閾値(score_threshold)は、現在「足切りライン」を探るため一時的に0.0（全通し）に設定。
        ログのスコアを見て、あとで 0.82 など適切な数値に変更。
        """
        used_texts = []
        used_scores = []
        used_past_replies = []

        # ==========================================
        # 1. 記憶の検索とスコア計算
        # ==========================================
        if self.memories and self.memory_embeddings is not None:
            query_emb = self.embed_model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, self.memory_embeddings)[0]
            
            scored_memories = []
            for i, score in enumerate(cos_scores):
                # ★ 4軸検索などで「特定の相手(current_speaker)」に絞る場合はここでif文を追加します
                memory = self.memories[i]
                scored_memories.append({
                    "text": memory.get("text", ""),
                    "past_reply": memory.get("past_reply", ""),
                    "score": score.item()
                })
            
            # スコアが高い順にソート
            scored_memories = sorted(scored_memories, key=lambda x: x["score"], reverse=True)
            
            # 閾値(score_threshold)以上のものだけを残し、上位 top_k 件を取得
            valid_memories = [m for m in scored_memories if m["score"] >= score_threshold][:top_k]
            
            used_texts = [m["text"] for m in valid_memories]
            used_scores = [f"{m['score']:.4f}" for m in valid_memories]
            used_past_replies = [m["past_reply"] for m in valid_memories if m.get("past_reply")]

        # ==========================================
        # 2. プロンプトの動的ルーティング
        # ==========================================
        if len(used_texts) > 0:
            # 【パターンA】関連する記憶が見つかった場合
            memory_context = "\n".join([f"- {t}" for t in used_texts])
            past_reply_context = "\n".join([f"- {r}" for r in used_past_replies if r])
            
            system_prompt = (
                "あなたは日本人です。必ず自然な日本語のみで返答してください。\n"
                "あなたはユーザーの友人としてLINEで会話をしています。\n"
                "以下の過去の記憶（文脈）と、あなたの過去の返答例を参考にして、相手のメッセージに自然に返信してください。\n\n"
                f"【過去の記憶】\n{memory_context}\n\n"
                f"【あなたの過去の返答例】\n{past_reply_context}\n\n"
                "指示:\n"
                "- LINEらしい短くカジュアルな口調を守ること。\n"
                "- AIのような丁寧すぎる言葉遣い（「〜ですね」「私は〜」など）は禁止。\n"
                "- 記憶にないことは適当にでっち上げず、自然に会話を繋ぐこと。"
            )
        else:
            # 【パターンB】記憶がない（足切りされた）場合
            system_prompt = (
                "あなたは日本人です。必ず自然な日本語のみで返答してください。\n"
                "あなたはユーザーの友人としてLINEで会話をしています。\n"
                "現在、相手の話題に関する過去の記憶がありません。\n\n"
                "指示:\n"
                "- LINEらしい短くカジュアルな口調を守ること。\n"
                "- AIのような丁寧すぎる言葉遣い（「〜ですね」「私は〜」など）は禁止。\n"
                "- 話題がわからない場合は、「ごめん、それいつの話だっけ？」「ちょっと覚えてないかも」と自然に聞き返すか、適当に相槌を打つこと。"
            )

        # ==========================================
        # 3. モデルが理解できるChatML形式へ変換
        # ==========================================
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        text_for_model = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_for_model], return_tensors="pt").to(self.model.device)

        # ==========================================
        # 4. 返答の生成（ストップトークン漏れ対策済み）
        # ==========================================
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
                eos_token_id=terminators # ここでハルシネーション（文字列漏れ）を強制ストップ
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # ==========================================
        # 5. 後処理
        # ==========================================
        reply = reply.replace("forgettable_id_", "").strip()
        reply = reply.split("\n")[0]  # LINEを想定し、余計な改行以降はカットする

        return {
            "reply": reply,
            "used_memories_count": len(used_texts),
            "used_memories_texts": used_texts,
            "used_memories_scores": used_scores
        }