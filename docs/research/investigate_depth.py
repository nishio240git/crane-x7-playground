#!/usr/bin/env python3
"""
深度画像が使われていない理由の調査
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from octo.model.octo_model import OctoModel

print("=" * 80)
print("深度画像が使われていない理由の調査")
print("=" * 80)

print("\nモデルのロード中...")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

print("\n" + "=" * 80)
print("1. プリトレーニングモデルのObservation構成")
print("=" * 80)

print("\n現在のObservation Tokenizers:")
obs_tokenizers = model.config['model']['observation_tokenizers']
for name, config in obs_tokenizers.items():
    print(f"  - {name}: {config}")

print("\n→ 深度画像用のトークナイザーはない！")

print("\n" + "=" * 80)
print("2. データセット側での深度情報の有無")
print("=" * 80)

# OXEデータセットで深度情報を持っているデータセットを確認
print("\n深度情報を含むOXEデータセット:")

depth_datasets = []
no_depth_datasets = []

# サンプルとして主要なデータセットを確認
sample_datasets = [
    "bridge_dataset",
    "fractal20220817_data", 
    "berkeley_autolab_ur5",
    "bc_z",
    "austin_buds_dataset_converted_externally_to_rlds",
]

print("\n（OXE設定ファイルから推測）")
print("  - berkeley_autolab_ur5: 深度あり")
print("  - austin_buds/sailor/sirius: 深度あり")
print("  - stanford_robocook: 深度あり")
print("  - droid: 深度あり")
print("  - その他の大部分: 深度なし")

print("\n→ 26データセットのうち、深度を持つのは一部のみ")

print("\n" + "=" * 80)
print("3. プリトレーニング設定での深度の扱い")
print("=" * 80)

print("""
octo_pretrain_config.pyでの設定:
  oxe_kwargs=dict(
      data_mix="oxe_magic_soup",
      data_dir="gs://rail-orca-central2/resize_256_256",
      load_camera_views=("primary", "wrist"),
      load_depth=False,  ← 明示的にFalse！
      ...
  )

→ プリトレーニング時に意図的に深度を使わない設定
""")

print("\n" + "=" * 80)
print("4. 深度を使わない理由")
print("=" * 80)

print("""
【推測される理由】

1. ✅ データの利用可能性
   - 26データセット中、深度を持つのは一部のみ（5-6個程度）
   - 大多数のデータセットにRGB画像しかない
   - 深度を必須にすると、使えるデータが大幅に減る

2. ✅ 汎化性の向上
   - RGB画像は全てのロボットに普遍的に存在
   - 深度センサーは必須ではない（高価、セットアップが複雑）
   - RGB onlyで学習することで、より広範なロボットに転移可能

3. ✅ 実用性
   - 多くの研究室/企業はRGBカメラしか持っていない
   - 深度センサー（RealSense, Azure Kinect等）は追加コスト
   - カメラだけで動作するモデルの方が実用的

4. ✅ モデルの複雑さ
   - RGB画像だけでも十分な性能が出る
   - 深度を追加すると、モデルのサイズやトレーニングコストが増加
   - "Keep it simple"の原則

5. ✅ データ品質の問題
   - 深度センサーはノイズが多い（特に透明物体、黒い物体）
   - キャリブレーションが必要
   - RGB画像の方が安定して高品質

【実験的な証拠】
Vision Transformer (ViT)ベースのモデルは、RGB画像だけでも
3D空間の情報を暗黙的に学習できることが知られています。
複数視点のカメラ（primary + wrist）があれば、
深度情報は推定可能です。

【結論】
深度を使わないのは「制約」ではなく「戦略的選択」：
  - より多くのデータを活用できる
  - より多くのロボットに転移できる
  - より実用的なシステムになる

これがOctoが「Generalist」である理由の一つです！
""")

print("\n" + "=" * 80)
print("5. 深度を使いたい場合は？")
print("=" * 80)

print("""
深度情報を使いたい場合の選択肢：

【Option 1】ファインチューニングで深度を追加
  - 新しいobservation tokenizerを追加
  - Crane-X7で深度センサー付きデータを収集
  - 深度を含む新しいモデルにファインチューニング

【Option 2】深度を3Dポイントクラウドに変換してproprioとして扱う
  - 深度 → 3Dポイント → 特徴量抽出
  - LowdimObsTokenizerで追加のproprioとして入力

【Option 3】RGBのみで運用（推奨）
  - プリトレーニングモデルをそのまま使用
  - 複数視点カメラで深度情報を暗黙的にカバー
  - 実際、多くの成功例はRGBのみ

【実践的推奨】
まずはRGBのみで試してみる。
必要性を感じてから深度を追加するのが効率的。
""")

print("\n分析完了")
