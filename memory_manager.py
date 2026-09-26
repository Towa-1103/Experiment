from datetime import datetime
import json
import os

class MemoryManager:

    def __init__(self, filepath="memories.json"):
        self.filepath = filepath
        self.memories = self._load()

    def _load(self):
        #memories.jsonを読み込む
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath)==0:
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def is_duplicate(self, new_item):
      """同一日時かつ同一テキストの場合のみ重複とみなす"""
      for existing in self.memories:
        # 日時だけでなく、要約内容（text）まで一致しているかチェック
        same_time = existing.get("timestamp") == new_item.get("timestamp")
        same_text = existing.get("text") == new_item.get("text")

        if same_time and same_text:
          return True
      return False

    def add_memories(self, new_memories):
      added_count = 0
      for item in new_memories:
        if self.is_duplicate(item):
          continue

        item["id"] = self._get_next_id()
        item["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.memories.append(item)
        added_count += 1

      self._save()
      return added_count

    def _save(self):
        #memories.jsonに書き出し
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def get_all(self):
        return self.memories