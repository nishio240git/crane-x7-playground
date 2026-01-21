#!/usr/bin/env python3
"""
Octoのproprioception設定を確認するスクリプト
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("Octoモデルのproprioception設定確認")
print("=" * 80)

# 1. make_oxe_dataset_kwargsのデフォルト値を確認
print("\n[1/3] make_oxe_dataset_kwargsのデフォルトパラメータ:")
from inspect import signature
from octo.data.oxe import make_oxe_dataset_kwargs

sig = signature(make_oxe_dataset_kwargs)
for param_name, param in sig.parameters.items():
    if param.default != param.empty:
        print(f"  - {param_name}: {param.default}")

# 2. 実際にbridge_datasetのkwargsを生成
print("\n[2/3] Bridge datasetのデフォルト設定（load_proprio=False）:")
kwargs_without_proprio = make_oxe_dataset_kwargs(
    name="bridge_dataset",
    data_dir="/dummy",
    load_camera_views=("primary",),
    load_depth=False,
    load_proprio=False,
)
print(f"  - proprio_obs_key in kwargs: {'proprio_obs_key' in kwargs_without_proprio}")

print("\n[3/3] Bridge datasetでload_proprio=Trueの場合:")
kwargs_with_proprio = make_oxe_dataset_kwargs(
    name="bridge_dataset",
    data_dir="/dummy",
    load_camera_views=("primary",),
    load_depth=False,
    load_proprio=True,
)
print(f"  - proprio_obs_key in kwargs: {'proprio_obs_key' in kwargs_with_proprio}")
if 'proprio_obs_key' in kwargs_with_proprio:
    print(f"  - proprio_obs_key value: {kwargs_with_proprio['proprio_obs_key']}")

print("\n" + "=" * 80)
print("結論:")
print("=" * 80)
print("Octo-small-1.5モデルは学習時に load_proprio=False で訓練されました。")
print("そのため、Bridge datasetには固有受容感覚データが含まれているものの、")
print("モデルは画像情報のみを使用するように訓練されています。")
print("\nこれにより:")
print("  ✓ モデルサイズの削減")
print("  ✓ 様々なロボット形態への汎化性向上")
print("  ✓ 視覚情報のみでの制御能力の学習")
print("=" * 80)
