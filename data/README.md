# Data

This project uses the **Deceptive Opinion Spam Corpus** (Ott et al., 2011): 1,600 hotel reviews,
800 truthful and 800 deceptive (written by paid crowd workers).

The corpus is **not redistributed in this repository**. Download it from the authors' page
(<https://myleott.com/op-spam.html>) or a public mirror of the combined CSV, then place it at:

```
data/raw/ott_deceptive_reviews.csv
```

Required columns: `text` and a label column. The label column may be called `label` or `deceptive`
(the name used in the original CSV), with values `truthful` / `deceptive` or `0` / `1`.
Other columns (`hotel`, `polarity`, `source`) are ignored.

`python scripts/prepare_data.py` then:

1. drops empty and duplicate review texts,
2. normalises the text,
3. writes a stratified 70 / 15 / 15 train / validation / test split (seed 42) to `data/splits/`.

**Citation:** Ott, M., Choi, Y., Cardie, C. and Hancock, J. T. (2011) *Finding Deceptive Opinion Spam
by Any Stretch of the Imagination*. Proceedings of ACL 2011.
