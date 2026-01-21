#!/usr/bin/env python3
"""
データセット統計の詳細分析 - サンプリング周期と標準偏差の関係
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
from octo.model.octo_model import OctoModel

print("=" * 80)
print("データセット統計の詳細分析")
print("=" * 80)

# モデルのロード
print("\nモデルのロード中...")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# 比較対象のデータセット
datasets_to_compare = [
    "bridge_dataset",
    "fractal20220817_data",
    "berkeley_autolab_ur5",
    "bc_z",
]

print("\n" + "=" * 80)
print("各データセットのアクション統計（XYZ移動）の比較")
print("=" * 80)

for dataset_name in datasets_to_compare:
    if dataset_name in model.dataset_statistics:
        stats = model.dataset_statistics[dataset_name]["action"]
        mean = stats["mean"]
        std = stats["std"]
        
        print(f"\n【{dataset_name}】")
        print(f"  Mean (XYZ): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
        print(f"  Std  (XYZ): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
        print(f"  → 標準偏差が大きい = 1ステップあたりの移動量のばらつきが大きい")
        
        # 軌跡数と遷移数を表示
        if "num_trajectories" in model.dataset_statistics[dataset_name]:
            num_traj = model.dataset_statistics[dataset_name]["num_trajectories"]
            num_trans = model.dataset_statistics[dataset_name]["num_transitions"]
            avg_len = num_trans / num_traj if num_traj > 0 else 0
            print(f"  軌跡数: {num_traj}, 遷移数: {num_trans}")
            print(f"  平均エピソード長: {avg_len:.1f} steps")

print("\n" + "=" * 80)
print("標準偏差の比較（X軸の例）")
print("=" * 80)

bridge_std_x = model.dataset_statistics["bridge_dataset"]["action"]["std"][0]
fractal_std_x = model.dataset_statistics["fractal20220817_data"]["action"]["std"][0]

print(f"\nBridge dataset    - X軸 std: {bridge_std_x:.6f} m/step")
print(f"Fractal dataset   - X軸 std: {fractal_std_x:.6f} m/step")
print(f"比率: {fractal_std_x / bridge_std_x:.2f}倍")

print("\n【解釈】")
print("標準偏差が大きい = データ内でのステップあたりの移動量のばらつきが大きい")
print("これは以下の要因が考えられます：")
print("  1. ロボットの物理的な動作範囲が大きい（アームが長い、可動域が広い）")
print("  2. タスクが多様で、大きな移動と小さな移動が混在している")
print("  3. 制御周期が異なる（低周波数 = 1ステップで大きく移動）")

print("\n" + "=" * 80)
print("サンプリング周期との関係")
print("=" * 80)

print("""
【重要な前提】
あなたの理解は基本的に正しいですが、重要な注意点があります：

1. **制御周期が同じ場合**
   - 標準偏差が大きい = 速い動作、大きな移動が多い
   - 標準偏差が小さい = ゆっくりした動作、小さな移動が多い

2. **制御周期が異なる場合**
   - 10Hzのロボット: 1ステップ = 0.1秒 → 大きく移動できる
   - 30Hzのロボット: 1ステップ = 0.033秒 → 小さく移動
   - 同じ速度でも、周期が違えば1ステップあたりの移動量が変わる！

3. **OXEデータセットの実態**
   - データセットごとに制御周波数が異なる可能性が高い
   - Bridgeデータセット: 記録されている制御周波数を確認する必要がある
   - Fractalデータセット: 同様に確認が必要

【結論】
標準偏差の差は以下の複合的な要因を反映しています：
  ✓ ロボットの物理的な動作範囲の違い
  ✓ タスクの性質の違い
  ✓ 制御周波数/サンプリング周期の違い
  ✓ データ収集時の動作速度の違い

単純に「標準偏差が大きい = EEF移動量が大きい」とは言えますが、
それが「ロボットが速い」のか「周期が長い」のかは区別が必要です。
""")

# より詳細な統計分析
print("\n" + "=" * 80)
print("実際の正規化・非正規化の例")
print("=" * 80)

normalized_value = 0.5  # モデルが出力する正規化された値

print(f"\nモデル出力（正規化済み）: {normalized_value}")
print(f"\n各データセット統計で非正規化した結果（X軸移動量）:")

for dataset_name in datasets_to_compare:
    if dataset_name in model.dataset_statistics:
        stats = model.dataset_statistics[dataset_name]["action"]
        mean_x = stats["mean"][0]
        std_x = stats["std"][0]
        
        # 非正規化: value_real = value_normalized * std + mean
        real_value = normalized_value * std_x + mean_x
        
        print(f"  {dataset_name:30s}: {real_value:.6f} m = {real_value*1000:.2f} mm")

print("""
→ 同じモデル出力でも、データセット統計によって実際の移動量が大きく変わる！
→ これが正しいデータセット統計を選ぶことが重要な理由です。
""")

print("=" * 80)
print("分析完了")
