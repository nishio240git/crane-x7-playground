#!/usr/bin/env python3
"""
正規化統計の重要性を理解するためのスクリプト
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import jax
import numpy as np
from octo.model.octo_model import OctoModel

print("=" * 80)
print("正規化統計の重要性デモ")
print("=" * 80)

# モデルのロード
print("\n[1/4] モデルのロード中...")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print("✓ モデルのロード完了")

# 利用可能なデータセット統計を確認
print("\n[2/4] 利用可能なデータセット統計:")
print(f"データセット数: {len(model.dataset_statistics)}")
for i, dataset_name in enumerate(list(model.dataset_statistics.keys())[:5]):
    print(f"  {i+1}. {dataset_name}")
print(f"  ... (全{len(model.dataset_statistics)}個)")

# Bridge datasetの統計情報を詳しく見る
print("\n[3/4] Bridge datasetの正規化統計の詳細:")
bridge_stats = model.dataset_statistics["bridge_dataset"]
print(f"統計情報のキー: {list(bridge_stats.keys())}")

if "action" in bridge_stats:
    action_stats = bridge_stats["action"]
    print(f"\nAction統計:")
    print(f"  - Mean (平均): {action_stats['mean']}")
    print(f"  - Std (標準偏差): {action_stats['std']}")
    print(f"  - Mask (使用フラグ): {action_stats.get('mask', 'なし')}")

# 実際の推論で正規化統計の重要性を示す
print("\n[4/4] 正規化統計の重要性:")

# 正規化統計なしとありの違いを計算で示す
print("\n--- Bridge datasetのアクション統計を使った変換 ---")

# モデルの生出力を想定（正規化された値）
normalized_action = np.array([0.5, -0.3, 0.8, 0.0, 0.1, -0.2, 1.0])
print(f"モデルの生出力 (正規化済み): {normalized_action}")

# Bridge統計で非正規化
mean = bridge_stats["action"]["mean"]
std = bridge_stats["action"]["std"]
mask = bridge_stats["action"]["mask"]

unnormalized_action = np.where(
    mask,
    normalized_action * std + mean,
    normalized_action
)
print(f"\n非正規化後 (実際のロボットの値):")
print(f"  X移動: {unnormalized_action[0]:.6f} m")
print(f"  Y移動: {unnormalized_action[1]:.6f} m")
print(f"  Z移動: {unnormalized_action[2]:.6f} m")
print(f"  Roll:  {unnormalized_action[3]:.6f} rad")
print(f"  Pitch: {unnormalized_action[4]:.6f} rad")
print(f"  Yaw:   {unnormalized_action[5]:.6f} rad")
print(f"  Gripper: {unnormalized_action[6]:.6f} (mask=Falseなので正規化なし)")

# 比較: 別のデータセット統計を使った場合
print("\n--- 異なる統計を使うと結果が変わる例 ---")
# fractal20220817_dataのデータセット統計
fractal_dataset = "fractal20220817_data"
if fractal_dataset in model.dataset_statistics:
    fractal_stats = model.dataset_statistics[fractal_dataset]["action"]
    fractal_unnorm = np.where(
        fractal_stats["mask"],
        normalized_action * fractal_stats["std"] + fractal_stats["mean"],
        normalized_action
    )
    print(f"Bridge統計で非正規化: {unnormalized_action}")
    print(f"Fractal統計で非正規化: {fractal_unnorm}")
    print(f"差分: {np.abs(unnormalized_action - fractal_unnorm)}")
    print("\n→ 同じモデル出力でも、統計が違うと実際のアクションが大きく変わる！")

# 正規化の計算式を説明
print("\n" + "=" * 80)
print("正規化の仕組み:")
print("=" * 80)
print("""
学習時: action_normalized = (action_real - mean) / std
推論時: action_real = action_normalized * std + mean

【重要性】
1. モデルは正規化された空間（平均0、標準偏差1）で学習している
2. 実際のロボットには元のスケールのアクションが必要
3. データセットごとにアクション範囲が異なる：
   - WidowX (Bridge): 特定の動作範囲
   - Franka Panda: 別の動作範囲
   - ALOHA: さらに別の範囲（デュアルアーム、14次元）
4. 間違った統計を使うと：
   - 動作が極端に大きい/小さくなる
   - ロボットの可動範囲を超えてエラー
   - グリッパーの開閉が逆になる可能性

【推奨】
- Crane-X7で使う場合: 最も似たロボットの統計を選ぶ
  - bridge_dataset: WidowX (6DOF + gripper)
  - fractal20220817_data: Franka Panda (7DOF + gripper)
- または、少量のデータでファインチューニングして独自の統計を作成
""")

print("\n確認完了")
