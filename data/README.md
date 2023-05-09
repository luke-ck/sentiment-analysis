The data contained in `twitter-datasets.zip`

```
data/
├── README.md
├── sample_submission.csv
├── test_data.txt
├── train_neg_full.txt
├── train_neg.txt
├── train_pos_full.txt
└── train_pos.txt
```

Additionally there is

- `train_pos_tiny.txt`
- `train_neg_tiny.txt`

which is a smaller version of `train_pos.txt`/`train_neg.txt` respectively. It was created by running:

- `head --lines=100 train_pos.txt > train_pos_tiny.txt`
- `head --lines=100 train_neg.txt > train_neg_tiny.txt`
