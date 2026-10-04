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
        importance = memory.get("importance", 0.5)
        try: # float型として読み込む
            return float(importance)
        except ValueError:
            return 0.1

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
        score_threshold: float = 0.8, # 4軸専用の厳しい閾値にデフォルトを変更
        scoring_method: str = "4-axis", 
        scoring_weights: dict = None
    ) -> dict:
        
        if scoring_weights is None:
            scoring_weights = {"relevance": 0.65, "recency": 0.1, "importance": 0.1, "sender_id": 0.15} # 話者の重みを0.3に調整

        used_texts = []
        used_scores = []

        # --- 1. 記憶の検索と足切り ---
        if self.memories and self.memory_embeddings is not None:
            query_emb = self.embed_model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, self.memory_embeddings)[0]
            
            scored_memories = []
            for i, memory in enumerate(self.memories):
                raw_relevance = cos_scores[i].item()
                
                # --- フィルター処理（足切り・正規化・3乗）を追加 ---
                baseline = 0.80
                if raw_relevance <= baseline:
                    norm_relevance = 0.0
                else:
                    norm_relevance = (raw_relevance - baseline) / (1.0 - baseline)
                    norm_relevance = min(1.0, norm_relevance)
                    
                # 変換後のスコアを関連度として重み掛け算
                final_score = norm_relevance * scoring_weights.get("relevance", 1.0)
                
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

        # --- 2. ルールの取得 ---
        rule = self.person_rules.get(current_speaker, self.person_rules.get("default"))
        relation = rule.get("relation", "不明")
        tone_rules = rule.get("tone_rules", "自然な日本語で返答してください。")

        # --- 3. システムプロンプトの構築 ---
        system_base = (
            "あなたは日本人です。必ず自然な日本語のみを使用して返答してください。中国語（簡体字・繁体字）や英語などの他言語は【絶対に】混入させないでください。\n"
            "あなたはLINEのチャットボットとしてユーザーと対話します。\n"
            "以下の【相手との関係性と口調の絶対ルール】を常に厳守してください。\n\n"
            f"【相手との関係性と口調の絶対ルール】\n"
            f"相手: {current_speaker}\n"
            f"関係性: {relation}\n"
            f"口調の指示: {tone_rules}\n"
        )

        if len(used_texts) > 0:
            # 記憶がヒットした場合（通常のRAG挙動）
            memory_context = "\n".join([f"- {t}" for t in used_texts])
            system_prompt = (
                system_base +
                f"\n【過去の記憶】\n{memory_context}\n\n"
                "指示:\n"
                "- 上記の記憶を踏まえて、話題に自然に返信してください。\n"
                "- 記憶に書かれていない具体的な事実や予定を勝手にでっち上げないでください。"
            )
        else:
            # 記憶が0件の場合：Few-Shotによる一般知識の強制遮断
            system_prompt = (
                system_base +
                "\n【重要指示: 記憶が存在しない場合の制約】\n"
                "現在、あなたと相手の間には、この話題に関する共通の記憶や過去のやり取りが一切ありません。\n"
                "相手からの質問に対して、自身の一般的な知識（実在する映画名、ゲーム名、店舗名、企業名など）を使った回答や提案は【一切禁止】します。\n"
                "過去を共有していないこと、または直近の状況を知らないことを率直に伝え、相手に聞き返す返答のみを行ってください。\n\n"
                "【返答の手本（以下の形式とトーンを厳守してください）】\n"
                "相手「最近おすすめの映画ある？」\n"
                "あなた「最近全然映画観てないから分からないな！何か面白いのあった？」\n"
                "相手「おすすめのゲーム教えて」\n"
                "あなた「最近新しいゲーム追えてないかも。何かハマってるのあるの？」\n"
                "相手「この前のトラブルどうなった？」\n"
                "あなた「すみません、その件については詳しく把握できておらず…どのような状況でしょうか？」\n"
                "相手「〇〇企業はどうだった？」\n"
                "あなた「あ、その企業についてはちょっと分からなくて。どうしたの？」\n"
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