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
python predict.py submissions/best_score_lb0.577.csv submissions/chris_sub.csv -w 2 -w 1 --ensemble-type rank-weighting
```
出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['submissions/best_score_lb0.577.csv', 'submissions/chris_sub.csv']
weights                    =  [2.0, 1.0]
ensemble_type              =  rank-weighting
n_top                      =  None
output                     =  rank-weighting_submission.csv

(1/5) Reading submission csv
Reading the given submission files: 100%|██████████| 2/2 [00:32<00:00, 16.31s/it]
Adding rank-weight col: 100%|██████████| 2/2 [00:28<00:00, 14.20s/it]
Checking records-length consistency of the given submissions: 100%|██████████| 2/2 [00:14<00:00,  7.06s/it]
Checking session_type consistency of the given submissions: 100%|██████████| 1/1 [02:38<00:00, 158.48s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|██████████| 1/1 [02:04<00:00, 124.12s/it]
(3/5) Calculating total rank_weight
(4/5) Creating submission-format csv
(5/5) Exporting as rank-weighting_submission.csv
```


### `ver009_order_recall_065736.csv` (LB 0.580) + `ver010_order_recall_06618.csv`（LB 0.584）
#### 重みが1:1
結果はLB 0.xxx。
```bash
python predict.py submissions/ver009_order_recall_065736.csv submissions/ver010_order_recall_06618.csv --ensemble-type rank-weighting
```
出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['submissions/ver009_order_recall_065736.csv', 'submissions/ver010_order_recall_06618.csv']
weights                    =  [1, 1]
ensemble_type              =  rank-weighting
n_top                      =  None

(1/5) Reading submission csv
Reading the given submission files: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:35<00:00, 17.55s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [01:42<00:00, 102.66s/it]
(3/5) Calculating total rank_weight
(4/5) Creating submission-format csv
(5/5) Exporting as rank-weighting_submission.csv
```
#### 重みが0.580:0.584
結果はLB 0.xxx。
```bash
python predict.py submissions/ver009_order_recall_065736.csv submissions/ver010_order_recall_06618.csv -w 0.580 -w 0.584 -e rank-weighting
```
出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['submissions/ver009_order_recall_065736.csv', 'submissions/ver010_order_recall_06618.csv']
weights                    =  [0.58, 0.584]
ensemble_type              =  rank-weighting
n_top                      =  None

(1/5) Reading submission csv
Reading the given submission files: 100%|█████████████████████████████████████████████████████████████| 2/2 [00:34<00:00, 17.12s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [01:42<00:00, 102.53s/it]
(3/5) Calculating total rank_weight
(4/5) Creating submission-format csv
(5/5) Exporting as rank-weighting_submission.csv
```


### CV計算用
```bash
python predict.py predicted_for_orders/orders_036_cv0.65736.csv predicted_for_orders/orders_044_cv0.66184.csv -e rank-weighting --n-top 40 -o rank-weighting_predicted.csv
```
出力例：
```text
* Given Parameters:
submission_csv_paths       =  ['predicted_for_orders/orders_036_cv0.65736.csv', 'predicted_for_orders/orders_044_cv0.66184.csv']
weights                    =  [1, 1]
ensemble_type              =  rank-weighting
n_top                      =  40
output                     =  rank-weighting_predicted.csv

(1/5) Reading submission csv
Reading the given submission files: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:39<00:00, 19.72s/it]
Adding rank-weight col: 100%|████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:16<00:00,  8.03s/it]
Checking records-length consistency of the given submissions: 100%|██████████████████████████████████████████████| 2/2 [00:10<00:00,  5.26s/it]
Checking session_type consistency of the given submissions: 100%|███████████████████████████████████████████████| 1/1 [04:45<00:00, 285.66s/it]
(2/5) Outer-join operation
Outer-join operating: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:53<00:00, 113.58s/it]
(3/5) Calculating total rank_weight
(4/5) Creating submission-format csv
predict.py:147: UserWarning: 
        if the given submission is for submissions, submission file must have 5015409 rows.

  warnings.warn("""
(5/5) Exporting as rank-weighting_predicted.csv
```
