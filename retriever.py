from datetime import datetime
import json
import math
import numpy as np

class MemoryRetriever:

  def __init__(self,embed_model, memory_file="memories.json"):
    self.memory_file = memory_file
    self.memories = self._load_memories()
    self.embeddings_cache = {}
    
    # ローカルのEmbeddingモデルを初期化（初回のみダウンロード）
    self.model = embed_model

  def _load_memories(self):
    try:
      with open(self.memory_file, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return []

  def _get_embedding(self, text: str) -> np.ndarray:
    if text in self.embeddings_cache:
      return self.embeddings_cache[text]

    # e5モデルの推奨フォーマット（クエリ用）
    # ※記憶側のテキストも同じ形式でベクトル化して問題ありません
    formatted_text = f"query: {text}"
    
    # Sentence-Transformersによるベクトル化
    vec = self.model.encode(formatted_text, normalize_embeddings=True)
    
    # numpy配列として保存（float32）
    vec = np.array(vec, dtype=np.float32)
    self.embeddings_cache[text] = vec
    return vec

  def _calc_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2))

  def _calc_recency_score(self, timestamp_str: str, decay_rate: float = 0.01) -> float:
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
    """
    mode="4axis": 提案手法。話者(0.30)を加味し、相手との文脈を維持する。
    mode="3axis": 既存RAGのベースライン。話者(0.0)を無視し、意味・鮮度・重要度だけで検索。
    """
    if not self.memories:
      return []

    if mode == "4axis":
      weights = {"semantic": 0.40, "speaker": 0.30, "recency": 0.15, "importance": 0.15}
    else:
      weights = {"semantic": 0.50, "speaker": 0.0, "recency": 0.25, "importance": 0.25}

    query_vec = self._get_embedding(query)
    scored_results = []

    for item in self.memories:
      text_vec = self._get_embedding(item.get("text", ""))
      sim_score = max(0.0, self._calc_cosine_similarity(query_vec, text_vec))
      
      speaker_score = 1.0 if item.get("sender_id") == current_speaker else 0.0
      recency_score = self._calc_recency_score(item.get("timestamp", ""))
      importance_score = min(max(item.get("importance", 5) / 10.0, 0.1), 1.0)

      total_score = (
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