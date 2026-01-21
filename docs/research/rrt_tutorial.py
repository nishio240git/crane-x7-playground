#!/usr/bin/env python3
"""
RRT（Rapidly-exploring Random Tree）の初学者向け解説
"""

print("=" * 80)
print("RRT（Rapidly-exploring Random Tree）超入門")
print("=" * 80)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. RRTとは？（直感的な理解）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【迷路を解くゲーム】

スタート S から ゴール G までの道を探す：

  ┌─────────────┐
  │ S   ■       │  S: スタート
  │    ■■       │  G: ゴール
  │     ■   G   │  ■: 壁（通れない）
  │    ■        │
  └─────────────┘

【RRTの戦略】
  1. スタート地点から始める
  2. ランダムな方向に少し進む（木を成長させる）
  3. 壁にぶつからないか確認
  4. OKなら、その点を記録
  5. ゴールに着くまで繰り返す

【アニメーション（イメージ）】

ステップ1:
  S

ステップ2:
  S
   └─・  ← ランダムに伸ばす

ステップ3:
  S
   ├─・
   └─・  ← もう1本

ステップ4:
  S
   ├─・─・
   └─・  ← 枝分かれして成長

ステップ10:
  S
   ├─・─・─・
   │   └─・
   └─・─・
       └─・─G  ← ゴールに到達！

この「木（Tree）」が空間を「素早く探索（Rapidly-exploring）」するので
RRT = Rapidly-exploring Random Tree

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. RRTのアルゴリズム（ステップバイステップ）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【準備】
  - Tree = {スタート地点}
  - ゴール地点を決める

【ループ（ゴールに着くまで繰り返し）】

  ステップ1: ランダムな点を選ぶ
  ┌─────────────┐
  │ S   ■    ×  │  ← × がランダムな点
  │    ■■       │
  │     ■   G   │
  │    ■        │
  └─────────────┘

  ステップ2: 木の中で最も近い点を探す
  ┌─────────────┐
  │ S   ■    ×  │
  │  ・ ■■       │  ← この・が最も近い
  │     ■   G   │
  │    ■        │
  └─────────────┘

  ステップ3: その点から、ランダムな点の方向に少し進む
  ┌─────────────┐
  │ S   ■    ×  │
  │  ・→NEW■       │  ← NEWが新しい点
  │     ■   G   │
  │    ■        │
  └─────────────┘

  ステップ4: 壁にぶつかるか確認
    - ぶつかる → この点は使えない、やり直し
    - ぶつからない → 木に追加！

  ステップ5: ゴールに近づいたか確認
    - 近づいた → 成功！経路が見つかった
    - まだ遠い → ステップ1に戻る

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. なぜRRTが優れているのか？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【比較: 全探索 vs RRT】

全探索（グリッド法）:
  すべてのマス目をチェック
  ┌─────────────┐
  │✓✓✓✓✓✓✓✓✓✓✓│  ← 全部チェック
  │✓✓✓✓✓✓✓✓✓✓✓│     時間がかかる！
  │✓✓✓✓✓✓✓✓✓✓✓│
  └─────────────┘
  計算量: O(n^d)  d=次元数
  7次元のロボット → 爆発的に増える

RRT:
  ランダムに探索
  ┌─────────────┐
  │ ・   ・      │  ← 必要な場所だけ
  │    ・  ・    │     効率的！
  │ ・      ・   │
  └─────────────┘
  計算量: はるかに少ない
  高次元でも動作可能

【RRTの強み】
  1. 高次元でも動作（ロボットの7関節 = 7次元OK）
  2. 複雑な障害物でも対応
  3. 実装が簡単
  4. 確率的完全性（時間をかければ必ず解が見つかる）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. 簡単なPython実装例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np

class RRT:
    def __init__(self, start, goal, obstacles):
        self.start = start    # スタート地点
        self.goal = goal      # ゴール地点
        self.obstacles = obstacles  # 障害物リスト
        self.tree = [start]   # 木（最初はスタートのみ）
        self.parent = {tuple(start): None}  # 親ノード記録

    def sample_random_point(self):
        \"\"\"ランダムな点を生成\"\"\"
        return np.random.rand(2) * 10  # 0-10の範囲

    def find_nearest_node(self, point):
        \"\"\"木の中で最も近いノードを探す\"\"\"
        distances = [np.linalg.norm(node - point) 
                     for node in self.tree]
        return self.tree[np.argmin(distances)]

    def steer(self, from_node, to_point, step_size=0.5):
        \"\"\"from_nodeからto_pointの方向にstep_size進む\"\"\"
        direction = to_point - from_node
        distance = np.linalg.norm(direction)
        if distance == 0:
            return from_node
        direction = direction / distance
        new_node = from_node + direction * min(step_size, distance)
        return new_node

    def is_collision_free(self, from_node, to_node):
        \"\"\"障害物にぶつからないか確認\"\"\"
        for obs in self.obstacles:
            # 簡略化: 円形障害物との衝突判定
            if np.linalg.norm(to_node - obs['center']) < obs['radius']:
                return False
        return True

    def plan(self, max_iterations=1000):
        \"\"\"経路計画の実行\"\"\"
        for i in range(max_iterations):
            # 1. ランダムな点をサンプル
            random_point = self.sample_random_point()
            
            # 2. 最も近いノードを探す
            nearest = self.find_nearest_node(random_point)
            
            # 3. その方向に進む
            new_node = self.steer(nearest, random_point)
            
            # 4. 衝突チェック
            if self.is_collision_free(nearest, new_node):
                # 5. 木に追加
                self.tree.append(new_node)
                self.parent[tuple(new_node)] = tuple(nearest)
                
                # 6. ゴールに到達したか確認
                if np.linalg.norm(new_node - self.goal) < 0.5:
                    print(f"ゴール到達！ {i}回のイテレーション")
                    return self.extract_path(new_node)
        
        return None  # 経路が見つからなかった

    def extract_path(self, goal_node):
        \"\"\"経路を抽出\"\"\"
        path = [goal_node]
        current = tuple(goal_node)
        while self.parent[current] is not None:
            current = self.parent[current]
            path.append(np.array(current))
        return path[::-1]  # 逆順にして返す


# 使用例
start = np.array([0, 0])
goal = np.array([9, 9])
obstacles = [
    {'center': np.array([5, 5]), 'radius': 1.5},
    {'center': np.array([3, 7]), 'radius': 1.0},
]

rrt = RRT(start, goal, obstacles)
path = rrt.plan()

if path:
    print("経路が見つかりました！")
    print(f"経路の長さ: {len(path)}点")
else:
    print("経路が見つかりませんでした")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. RRTの改良版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【RRT*（RRT Star）】
  RRTの改良版、より最適な経路を探す
  
  追加機能:
    - 新しいノードを追加するとき、周辺のノードも再接続
    - より短い経路があれば、親を変更
  
  結果:
    経路がどんどん改善される
    最適性保証（漸近的）

【Informed RRT*】
  さらに効率的
  
  改善:
    - ゴールを見つけた後、その経路長を使って探索範囲を制限
    - 無駄な場所を探索しない
  
  結果:
    より速く、より良い経路

【Bi-directional RRT】
  両側から探索
  
  戦略:
    - スタートから1本の木
    - ゴールからも1本の木
    - 2つの木が出会ったら成功
  
  結果:
    探索時間が半分に

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. 学習リソース
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【日本語書籍】

1. 「ロボティクス」 
   著者: 日本ロボット学会
   レベル: 入門〜中級
   内容: ロボット工学の基礎から軌道計画まで
   価格: ¥3,000-4,000
   
2. 「実践ロボット制御」
   著者: 大隅久, 池田毅
   レベル: 中級
   内容: 制御理論と実装
   
3. 「ROS2とPythonで作って学ぶAIロボット入門」
   著者: 出村公成
   レベル: 入門〜中級
   内容: ROSでの実装、MoveIt!の使い方

【英語書籍（推奨）】

1. "Planning Algorithms" by Steven M. LaValle ★★★★★
   http://lavalle.pl/planning/
   - 無料でオンラインで読める！
   - RRTの発明者による教科書
   - 最も詳しい解説
   - レベル: 中級〜上級

2. "Robotics: Modelling, Planning and Control"
   著者: Bruno Siciliano, Lorenzo Sciavicco
   - ロボット工学の総合的な教科書
   - 運動学、動力学、制御、計画すべて
   - レベル: 中級〜上級

3. "Principles of Robot Motion"
   著者: Howie Choset et al.
   - モーションプランニングに特化
   - 様々なアルゴリズムを網羅
   - レベル: 中級

【オンラインリソース】

1. Steven LaValle's Planning Algorithms
   http://lavalle.pl/planning/
   - 完全無料
   - Chapter 5がRRT
   - 最高の教材

2. YouTube: "Aaron Becker - RRT"
   - 可視化が素晴らしい
   - 直感的に理解できる

3. Wikipedia: RRT
   - 基本的な説明
   - アルゴリズムの擬似コード

4. Python Robotics
   https://github.com/AtsushiSakai/PythonRobotics
   - RRTの実装例多数
   - 可視化付き
   - 日本人作者

【論文（原典）】

1. LaValle, S. M., & Kuffner, J. J. (2001)
   "Rapidly-Exploring Random Trees: Progress and Prospects"
   - RRTの原論文
   - 読みやすい

2. Karaman, S., & Frazzoli, E. (2011)
   "Sampling-based algorithms for optimal motion planning"
   - RRT*の論文
   - より高度

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. 学習ステップ（推奨カリキュラム）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Phase 1: 基礎準備（1-2週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ Python基礎（既に知っていればスキップ）
  □ NumPy（配列操作、ベクトル計算）
  □ Matplotlib（グラフ描画）
  
  実践:
    - 2Dの点や線を描画してみる
    - ユークリッド距離を計算してみる

【Phase 2: RRTの理解（1週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ Steven LaValle's Planning Algorithms Chapter 5.1-5.5を読む
  □ YouTubeで可視化動画を見る
  □ Wikipediaのアルゴリズム擬似コードを理解
  
  課題:
    - 紙に手書きでRRTのステップを描いてみる
    - 2D空間での簡単な例を追ってみる

【Phase 3: 基本実装（1-2週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ 2D空間でのRRT実装
  □ 障害物なしで動作確認
  □ 円形障害物を追加
  □ 矩形障害物を追加
  
  実装順序:
    1. ツリー構造（リスト + 親辞書）
    2. ランダムサンプリング
    3. 最近傍探索
    4. ステアリング関数
    5. 衝突判定
    6. 経路抽出
  
  参考:
    - Python Robotics のRRT実装をコピーして動かす
    - 自分で書き直してみる

【Phase 4: 可視化（1週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ Matplotlibで木の成長を可視化
  □ アニメーション作成
  □ 最終経路のハイライト
  
  実装:
    - 各イテレーションで図を更新
    - GIFアニメーション保存

【Phase 5: 改良版（2週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ RRT* の実装
    - rewire機能の追加
    - 経路コストの計算
  □ Goal biasing（ゴール方向への偏り）
  □ Bi-directional RRT
  
  学習:
    - 各改良がどう性能を向上させるか実験

【Phase 6: 3D・高次元（2週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ 3D空間への拡張
  □ ロボットアームの関節空間（7次元）
  □ より複雑な障害物
  
  課題:
    - Crane-X7の関節空間でRRT

【Phase 7: ROS統合（2週間）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ MoveIt!の理解
  □ OMPLライブラリの使用
  □ Crane-X7での実機テスト
  
  実践:
    - RVizでの可視化
    - 実機での動作確認

【Phase 8: 応用（継続）】
━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ 動的障害物への対応
  □ リアルタイム再計画
  □ 複数ロボットの調整
  □ 学習ベースの手法との組み合わせ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. 実践的なTips
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【デバッグのコツ】
  1. まず2Dで完璧に動かす
  2. 可視化を先に作る（問題が見える）
  3. 少ないイテレーション（10-100）で試す
  4. 単純な障害物から始める

【性能改善】
  1. KD-Treeで最近傍探索を高速化
  2. Goal biasingで収束を早める（10-20%の確率でゴール方向）
  3. ステップサイズを調整
  4. 最大イテレーション数を適切に設定

【よくあるミス】
  ✗ ステップサイズが大きすぎる → 障害物を突き抜ける
  ✗ ランダムサンプリングの範囲が狭い → 解が見つからない
  ✗ 衝突判定が甘い → 実際には衝突している
  ✗ 経路抽出のループが無限 → 親ノードの管理ミス

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
まとめ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RRT = ランダムに木を成長させて経路を探す

【学習の鍵】
  1. Steven LaValle's Planning Algorithms（無料！）
  2. Python Roboticsで実装を見る
  3. 2Dから始めて、徐々に高次元へ
  4. 可視化して理解を深める
  5. 自分で実装してみる（写すだけでなく）

【学習期間の目安】
  - 基本理解: 1-2週間
  - 実装: 2-4週間
  - 応用: 1-3ヶ月
  - 計: 2-4ヶ月で実用レベル

【次のステップ】
  RRT → RRT* → Informed RRT* → 学習ベース手法（Octoなど）

頑張ってください！
""")

print("\n分析完了")
