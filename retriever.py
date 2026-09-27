from datetime import datetime
import json
import math
from google import genai
import numpy as np


class MemoryRetriever:

  def __init__(self, memory_file="memories.json", api_key=None):
    self.memory_file = memory_file
    self.client = genai.Client(api_key=api_key) if api_key else None
    self.memories = self._load_memories()
    self.embeddings_cache = {}

  def _load_memories(self): # memories.jsonを読込
    try:
      with open(self.memory_file, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return []

  def _get_embedding(self, text: str) -> np.ndarray: #テキストをGeminiに送り、数値ベクトルに変換
    if text in self.embeddings_cache: # 既にキャッシュにあればAPIを使わずそれを返す
      return self.embeddings_cache[text]

    response = self.client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    vec = np.array(response.embedding.values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
      vec = vec / norm
    self.embeddings_cache[text] = vec
    return vec

  def _calc_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float: #Semantic(コサイン類似度)を計算
    return float(np.dot(vec1, vec2))

  def _calc_recency_score(self, timestamp_str: str, decay_rate: float = 0.01) -> float: # Recencyを計算
    try:
      mem_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
      try:
        mem_date = datetime.strptime(timestamp_str, "%Y-%m-%d")
      except ValueError:
        return 0.5

    delta_days = (datetime.now() - mem_date).total_seconds() / 86400.0
    if delta_days < 0:
      delta_days = 0
    return math.exp(-decay_rate * delta_days)

  def search(self, query: str, current_speaker: str, top_k: int = 3, mode: str = "4axis"): 
    # 計算したSemantic, Recency, 設定されてるSender_id, Importanceを用いて上位k件を計算
    """
    mode="4axis": 意味(0.40), 鮮度(0.15), 重要度(0.15), 話者(0.30) <- 提案手法
    mode="3axis": 意味(0.50), 鮮度(0.25), 重要度(0.25), 話者(0.0)  <- 既存RAGのベースライン
    """
    if not self.memories:
      return []

    # モードによる重みの切り替え
    if mode == "4axis":
      weights = {"semantic": 0.40, "speaker": 0.30, "recency": 0.15, "importance": 0.15}
    else:
      # 話者の重みを 0 にし、既存RAGのように意味・鮮度・重要度だけで検索する
      weights = {"semantic": 0.50, "speaker": 0.0, "recency": 0.25, "importance": 0.25}

    query_vec = self._get_embedding(query)
    scored_results = []

    for item in self.memories: # memories.josnの全記憶に対してループを回し、スコアを計算
      text_vec = self._get_embedding(item.get("text", "")) 
      sim_score = max(0.0, self._calc_cosine_similarity(query_vec, text_vec)) # クエリと記憶テキストの類似度
      
      speaker_score = 1.0 if item.get("sender_id") == current_speaker else 0.0 # 合致判定で 1 or 0
      recency_score = self._calc_recency_score(item.get("timestamp", ""))
      importance_score = min(max(item.get("importance", 5) / 10.0, 0.1), 1.0) # importanceの値を10で割り正規化

      total_score = ( # 総合スコア計算
          weights["semantic"] * sim_score
          + weights["speaker"] * speaker_score
          + weights["recency"] * recency_score
          + weights["importance"] * importance_score
      )

      scored_results.append({
          "memory": item,
          "total_score": round(total_score, 4),
          "breakdown": {
              "semantic": round(sim_score, 3),
              "speaker": round(speaker_score, 1),
              "recency": round(recency_score, 3),
              "importance": round(importance_score, 2),
          }
      })

    scored_results.sort(key=lambda x: x["total_score"], reverse=True)
    return scored_results[:top_k]