# Image Calculator

手書きの数式画像をSSDモデルで検出し、計算結果を返すプロジェクトです。

## ディレクトリ

- `src/image_calculator/`: 推論・計算の本体
- `scripts/`: データ生成、確認、推論などの実行用スクリプト
- `notebooks/`: 学習・実験用ノートブック
- `assets/fonts/`: データ生成に使うフォント
- `models/`: 学習済みモデル
- `data/generated/`: 生成途中・比較用データセット
- `data/processed/`: 学習・検証・テスト用データセット
- `data/archives/`: データセットのZIP保管場所
- `outputs/`: 学習ログ、チェックポイント、プレビュー

## 成功・失敗例の可視化

保存済みモデルの予測とVOCの正解を重ね、フォントごとに成功・失敗例を保存します。

```bash
uv run python scripts/visualize_predictions.py \
  --model outputs/logs/baseline/model/ssd_calculator_seed42_state_dict.pth \
  --characters 1 7 \
  --result all \
  --iou-threshold 0.85 \
  --score-threshold 0.3 \
  --num-images 30
```

Slurmで実行する場合は、プロジェクトルートから次のように投入します。Tesla GPUを1枚使い、ログを`outputs/slurm/`に保存します。

```bash
mkdir -p outputs/slurm
sbatch scripts/run_visualize_predictions.sbatch \
  --model outputs/logs/baseline/model/ssd_calculator_seed42_state_dict.pth \
  --characters 1 7 \
  --result all \
  --iou-threshold 0.85 \
  --score-threshold 0.3 \
  --num-images 30
```

`sbatch`のスクリプト名以降の引数は、そのまま`visualize_predictions.py`に渡します。省略時はPythonスクリプトの既定値（baselineのseed42モデルなど）を使います。

- `--font Caveat-Bold.ttf`を付けると、そのフォントだけを対象にします。省略時は各フォントから抽出します。
- `--result`は`all`（両方）・`success`・`failure`。`--num-images`は各フォント・各結果の上限です。条件に合う画像が足りなければ、見つかった枚数を保存します。
- `--characters`は注目する正解クラスです。省略時は全クラスを対象にし、画像には指定の有無によらず全クラスを描きます。
- `--image-ids image_797 image_100`でsplit内の画像IDを指定できます。`--seed`は画像を調べる順番を決め、モデルの学習seedとは独立です。

スコア下限以上の予測を対象に、同クラスかつIoUが閾値以上の対応をscore降順で確保します。残りはクラスを問わず、重なりがある枠をIoU降順で1対1対応付けします。対応した文字はクラス一致かつIoUが閾値以上なら成功、それ以外は失敗です。対応できない正解は`missed`、余分な予測は`duplicate`または`extra_prediction`として記録します。これは目視分析用の分類で、COCO APの評価処理とは別です。

画像は、対象文字の成功・失敗を一つでも含めば各グループに入ります。同じ画像が両方に保存される場合もあります。余分な予測は、予測クラスが対象か、対象の正解枠と閾値以上重なる場合に失敗例の抽出対象になります。

出力先の例は`outputs/logs/baseline/plots/seed42/Caveat-Bold/iou0.85_score0.3/success/`です。`--output-dir`で基点を変更できます。モデル名に`seed数字`があればそのseedを使用し、それ以外はモデルのファイル名をディレクトリ名に使います。

各グループに次のファイルを保存します。

- 画像IDごとのPNG：正解は緑、予測は青。抽出対象の文字・予測には`*`を付け、画像下部にIoUと判定理由を表示します。
- `detections.csv`：保存した画像の文字・予測ごとの判定。`is_target`は注目対象、`selected`はそのグループへの抽出理由になった行を表します。余分な予測の`iou`は最も重なる正解とのIoUで、`nearest_gt_index`にその正解を記録します。
- `predictions.json`：保存した画像の全正解、モデル出力、判定、実行条件。描画スコア下限未満の予測も保存します（モデル内部のNMS等は適用済み）。

CSV/JSONは抽出した画像のみの記録です。成功・失敗を同じ枚数だけ集めるため、保存件数からvalidation全体の誤り率は計算できません。

