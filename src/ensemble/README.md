# Modified "💡 [2 methods] How-to ensemble predictions 🏅🏅🏅"

https://www.kaggle.com/code/radek1/2-methods-how-to-ensemble-predictions

留意点：
- GPUは一切使わない。
- pandasじゃなくてpolars使ってる。

# はじめに
```bash
python -m pip install -r requirements.txt
```

# 推論の実行例
## Voting
### `chris_sub.csv`（LB 0.567） + `ver008_cv_0554845.csv`（LB 0.562）
結果はLB 0.562。

```bash
time python predict.py chris_sub.csv ver008_cv_0554845.csv --ensemble-type voting --n-top 20
```
ここで、スコアが悪くなるため`--n-top 20`で読み込む各submission fileについて、session_type毎にtop 20のaidだけを取り出すように指定している。\
恐らくaidの順序関係なく重複するものを優先的に取り出すようにしているため、順位の低いものであっても重複すると選ばれやすくなるから悪くなる。\
なお、`--n-top 20`を指定しないときのスコアはLB 0.314であった。

出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['chris_sub.csv', 'ver008_cv_0554845.csv']
weights                    =  [1, 1]
ensemble_type              =  voting
n_top                      =  20

(1/5) Reading submission csv
Reading the given submission files: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:56<00:00, 28.03s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:17<00:00, 137.46s/it]
(3/5) Calculating total votes
(4/5) Creating submission-format csv
(5/5) Exporting as voting_submission.csv

real    8m22.410s
user    10m18.509s
sys     1m58.415s
```

## Rank-weighting
### `chris_sub.csv`（LB 0.567）+`ver008_cv_0554845.csv`（LB 0.562）
結果はLB 0.572。

```bash
time python predict.py chris_sub.csv ver008_cv_0554845.csv --ensemble-type rank-weighting
```
出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['chris_sub.csv', 'ver008_cv_0554845.csv']
weights                    =  [1, 1]
ensemble_type              =  rank-weighting
n_top                      =  None

(1/5) Reading submission csv
Reading the given submission files: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [01:02<00:00, 31.39s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:09<00:00, 129.52s/it]
(3/5) Calculating total rank_weight
(4/5) Creating submission-format csv
(5/5) Exporting as rank-weighting_submission.csv

real    8m40.976s
user    12m37.661s
sys     2m5.652s
```

### `best_score_lb0.577.csv` (LB 0.577) + `chris_sub.csv`（LB 0.567）
結果はLB 0.573。

```bash
python predict.py best_score_lb0.577.csv chris_sub.csv -w 2 -w 1 --ensemble-type rank-weighting
```
出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['best_score_lb0.577.csv', 'chris_sub.csv']
weights                    =  [2.0, 1.0]
ensemble_type              =  rank-weighting
n_top                      =  None

(1/5) Reading submission csv
Reading the given submission files: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:40<00:00, 20.32s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:25<00:00, 85.19s/it]
(3/5) Calculating total rank_weight
(4/5) Creating submission-format csv
(5/5) Exporting as rank-weighting_submission.csv
```