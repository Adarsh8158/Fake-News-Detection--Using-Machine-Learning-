# 📊 Dataset Information

This project is trained on the benchmark **ISOT Fake News Dataset** from the University of Victoria.

- **True.csv**: ~21,417 articles from verified Reuters world and politics reports.
- **Fake.csv**: ~23,481 articles from flagged unverified websites and conspiracy sources.

### Downloading Full Dataset
Because the full dataset files are over 115 MB combined, they are excluded from Git via `.gitignore`.
You can download the full dataset from:
- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `True.csv` and `Fake.csv` directly into this `data/` folder.

A lightweight `sample_news.csv` is provided in this folder for quick testing and batch predictions.
