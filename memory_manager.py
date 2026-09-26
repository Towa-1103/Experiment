from datetime import datetime
import json
import os


class MemoryManager:

  def __init__(self, file_path="memories.json"):
    self.file_path = file_path
    self.memories = self._load()

  def _load(self):
    if not os.path.exists(self.file_path):
      return []
    try:
      with open(self.file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return []

  def _save(self):
    with open(self.file_path, "w", encoding="utf-8") as f:
      json.dump(self.memories, f, ensure_ascii=False, indent=2)

  def _get_next_id(self):
    """既存データの最大ID + 1 を返す（データが空なら 1）"""
    if not self.memories:
      return 1
    return max(m.get("id", 0) for m in self.memories) + 1

  def is_duplicate(self, new_item):
    """日時と要約本文（text）が両方一致する場合のみ重複とみなす"""
    for existing in self.memories:
      same_time = existing.get("timestamp") == new_item.get("timestamp")
      same_text = existing.get("text") == new_item.get("text")
      if same_time and same_text:
        return True
    return False

  def add_memories(self, new_memories):
    added_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for item in new_memories:
      if self.is_duplicate(item):
        continue

      item["id"] = self._get_next_id()
      item["created_at"] = now_str
      self.memories.append(item)
      added_count += 1

    if added_count > 0:
      self._save()

    return added_count

  def get_all(self):
    return self.memories