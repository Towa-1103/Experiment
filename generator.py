from retriever import MemoryRetriever

class ResponseGenerator:
    # 外部（Colab）でロード済みの model と tokenizer を受け取る
    def __init__(self, model, tokenizer, memory_file="memories.json"):
        self.model = model
        self.tokenizer = tokenizer
        
        # Retrieverの初期化 (内部でSentenceTransformerのみロードされます)
        self.retriever = MemoryRetriever(memory_file=memory_file)
        
        self.score_threshold = 0.50
        self.max_memories = 2

    def _build_messages(self, current_speaker: str, query: str, memories: list) -> tuple:
        """Qwenに渡すシステムプロンプトとチャット履歴を構築する"""
        memory_texts = []
        
        for mem in memories:
            if mem['total_score'] >= self.score_threshold:
                m_data = mem['memory']
                memory_texts.append(
                    f"・日時: {m_data.get('timestamp')}\n"
                    f"  要約: {m_data.get('text')}\n"
                    f"  過去のあなたの返答例: {m_data.get('past_reply')}"
                )
            if len(memory_texts) >= self.max_memories:
                break
        
        memory_context = "\n\n".join(memory_texts) if memory_texts else "（関連する記憶は見つかりませんでした）"

        system_prompt = f"""あなたはLINEのユーザー本人として、相手({current_speaker})からのメッセージに返信してください。

【制約事項】
1. 以下の「過去の記憶」に、現在の会話と関連する情報があれば踏まえて返答してください。
2. 記憶が全く無関係（または「記憶なし」）の場合は、記憶のことは完全に無視して、自然に相槌や返答だけを行ってください。無理に話題に出してはいけません。
3. 返答はLINEのチャットらしく、短く、口語体で出力してください。
4. 「過去のあなたの返答例」がある場合、その口調やテンションを可能な限り模倣してください。

【過去の記憶】
{memory_context}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        return messages, memory_texts

    def generate_reply(self, query: str, current_speaker: str) -> dict:
        # 1. 4軸検索の実行
        search_results = self.retriever.search(
            query=query, 
            current_speaker=current_speaker, 
            top_k=3, 
            mode="4axis"
        )
        
        # 2. メッセージの構築
        messages, used_memories = self._build_messages(current_speaker, query, search_results)
        
        # 3. プロンプトのテンプレーティング
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 4. 返答の生成
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05
        )
        
        # 5. 出力テキストの抽出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "reply": reply.strip(),
            "used_memories_count": len(used_memories),
            "used_memories_texts": used_memories
        }