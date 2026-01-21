#!/usr/bin/env python3
"""
Octoモデルの仕様確認スクリプト
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("Octoモデルの仕様確認")
print("=" * 80)

print("\n[1/2] モデルのロード中...")
from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print("✓ モデルのロード完了")

print("\n[2/2] モデルの仕様を表示:")
print("-" * 80)
print(model.get_pretty_spec())
print("-" * 80)

# 追加情報の確認
print("\n追加情報:")
print(f"- 利用可能なデータセット統計: {list(model.dataset_statistics.keys())}")
print(f"- 設定されているobservation tokenizers: {list(model.config['model']['observation_tokenizers'].keys())}")
print(f"- 設定されているtask tokenizers: {list(model.config['model']['task_tokenizers'].keys())}")

# example_batchの構造を確認
print("\n観測データの構造:")
for key, value in model.example_batch["observation"].items():
    if hasattr(value, 'shape'):
        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  - {key}: type={type(value)}")

print("\nタスクデータの構造:")
for key, value in model.example_batch["task"].items():
    if key != "pad_mask_dict":
        if hasattr(value, 'shape'):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: type={type(value)}")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"    - {k}: shape={v.shape}, dtype={v.dtype}")

print("\n" + "=" * 80)
print("確認完了")
print("=" * 80)
