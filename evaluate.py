import csv
import os
from datetime import datetime

def run_experiment(
    bot, # 1.エンジン本体
    query: str, # 2.AIに投げるテキスト
    speaker: str, # 3.対話相手のID
    score_threshold: float = 0.83, # 4.類似度スコアの閾値
    top_k: int = 2, # 5.何件をプロンプトに渡すか
    scoring_method: str = "baseline", # 6.評価手法(3-axis, 4-axis)
    scoring_weights: dict = None, # 7.各要素の重み比率
    experiment_note: str = "テスト", # 8.自由記述欄
    csv_filename: str = "experiment_log_with_score.csv" # 9.出力先のCSVファイル名
):
    """
    RAGの検索条件を動的に変更しながらAIの生成結果をテストし、CSVにログを記録する関数。
    """
    if scoring_weights is None: # Noneの場合、類似度だけで返答生成
        scoring_weights = {"semantic": 1.0}

    print("-" * 50) 
    print(f"【実験条件】 手法: {scoring_method} | 閾値: {score_threshold} | メモ: {experiment_note}")
    print(f"相手({speaker}): {query}")

    # bot (ResponseGeneratorインスタンス) を使って生成
    result = bot.generate_reply(
        query=query, 
        current_speaker=speaker,
        top_k=top_k,
        score_threshold=score_threshold,
        scoring_method=scoring_method,   # 3軸か4軸かの指定を渡す
        scoring_weights=scoring_weights  # 重みの設定を渡す
    )

    # 生成の結果をそれぞれ格納
    reply = result['reply']
    used_count = result['used_memories_count']
    texts = result['used_memories_texts']
    scores = result['used_memories_scores']

    print(f"\nあなた(AI): {reply}\n")

    if used_count > 0:
        score_details = "\n".join([f"[Score: {score}] {text}" for text, score in zip(texts, scores)])
        print("【検索・使用された記憶】\n" + score_details)
    else:
        score_details = f"※該当する記憶なし（スコア {score_threshold} 未満のため足切り）"
        print("【検索・使用された記憶】\n" + score_details)
    print("-" * 50)

    # CSVへの記録
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Experiment_Note", "Scoring_Method", "Weights", "Threshold", 
                "Speaker", "Query", "AI_Reply", "Used_Memories_Count", "Score_Breakdown"
            ])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp, experiment_note, scoring_method, str(scoring_weights), score_threshold,
            speaker, query, reply, used_count, score_details
        ])

    return result