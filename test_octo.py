#!/usr/bin/env python3
"""
Octo動作確認スクリプト
"""

import sys

print("=" * 60)
print("Octo動作確認テスト")
print("=" * 60)

# 1. 基本的なインポート確認
print("\n[1/4] 基本的なインポート確認...")
try:
    import jax
    print(f"  ✓ JAX version: {jax.__version__}")
except ImportError as e:
    print(f"  ✗ JAXのインポートに失敗: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"  ✗ TensorFlowのインポートに失敗: {e}")
    sys.exit(1)

try:
    import flax
    print(f"  ✓ Flax version: {flax.__version__}")
except ImportError as e:
    print(f"  ✗ Flaxのインポートに失敗: {e}")
    sys.exit(1)

# 2. Octoモジュールのインポート確認
print("\n[2/4] Octoモジュールのインポート確認...")
try:
    import octo
    print(f"  ✓ Octo moduleをインポートしました")
except ImportError as e:
    print(f"  ✗ Octoのインポートに失敗: {e}")
    sys.exit(1)

try:
    from octo.model.octo_model import OctoModel
    print(f"  ✓ OctoModelをインポートしました")
except ImportError as e:
    print(f"  ✗ OctoModelのインポートに失敗: {e}")
    sys.exit(1)

# 3. JAXの動作確認
print("\n[3/4] JAXの動作確認...")
try:
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3, 4, 5])
    y = x * 2
    print(f"  ✓ JAX配列の計算: {x.tolist()} * 2 = {y.tolist()}")
except Exception as e:
    print(f"  ✗ JAXの動作確認に失敗: {e}")
    sys.exit(1)

# 4. Octo環境情報の表示
print("\n[4/4] Octo環境情報...")
try:
    import os
    octo_path = os.path.dirname(octo.__file__)
    print(f"  ✓ Octoパッケージパス: {octo_path}")
    
    # 利用可能なデバイスを確認
    devices = jax.devices()
    print(f"  ✓ 利用可能なJAXデバイス: {devices}")
    
except Exception as e:
    print(f"  ⚠ 環境情報の取得に失敗: {e}")

print("\n" + "=" * 60)
print("動作確認完了！")
print("=" * 60)
print("\n注意: モデルのダウンロードには別途実行が必要です。")
print("例: model = OctoModel.load_pretrained('hf://rail-berkeley/octo-small-1.5')")
print("=" * 60)
