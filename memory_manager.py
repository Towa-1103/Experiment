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
            with open(self.filepath, "r", encodeing="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def add_memories(self, new_items):
        #重複を排除しながら新しい記憶を追記保存する
        if not new_items:
            return 0

        existing_keys = { # timestampとtextの組み合わせが一致するものは除外
            (m.get("timestamp"), m.get("text")) for m in self.memories
        }

        current_max_id = max([m.get("id", 0) for m in self.memories], default=0)

        added_count = 0
        for item in new_items:
            key = (item.get("timestamp"), item.get("text"))
            if key not in existing_keys:
                current_max_id += 1
                item["id"] = current_max_id
                #追加日時を記録
                item["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                self.memories.append(item)
                existing_keys.add(key)
                added_count += 1

            if added_count > 0:
                self._save()

            return added_count

        def _save(self):
            #memories.jsonに書き出し
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)

        def get_all(self):
            return self.memories