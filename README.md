# Fake News Detection using Machine Learning

A TF-IDF + Logistic Regression classifier that labels news articles as **Real** or **Fake**, with word-level explanations of each prediction and a Streamlit app to try it out.

## Features

- Text preprocessing pipeline (`src/preprocess.py`)
- TF-IDF features + Logistic Regression classifier (`src/train.py`)
- Explainability: shows which words pushed a prediction toward Fake or Real, using the model coefficients (`src/explain.py`)
- Web app for live predictions (`app.py`)
- Robustness check for dataset leakage (`src/leakage_check.py`)
- Saved evaluation outputs in `reports/` (`metrics.json`, `confusion_matrix.png`)

## Project Structure

```
Fake-News-Detection--Using-Machine-Learning-/
├── data/                  # dataset files (not committed, see Dataset)
├── models/                # trained model files
├── reports/               # metrics.json, confusion_matrix.png, leakage_check_results.json
├── src/
│   ├── preprocess.py      # text cleaning
│   ├── train.py           # training and evaluation
│   ├── explain.py         # word-level explanations
│   └── leakage_check.py   # robustness check (see below)
├── tests/                 # unit tests
├── 01_NLP_Basics.ipynb    # notebook: NLP basics and experiments
├── app.py                 # web app
├── requirements.txt
├── run.bat                # Windows helper script
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Adarsh8158/Fake-News-Detection--Using-Machine-Learning-.git
cd Fake-News-Detection--Using-Machine-Learning-

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Dataset

The project uses the **ISOT Fake News dataset**, which consists of two files: `Fake.csv` and `True.csv`. Download it (search "ISOT Fake News Dataset" on Kaggle or the University of Victoria ISOT lab page) and place both files in the `data/` folder:

```
data/
├── Fake.csv
└── True.csv
```

The data files are not stored in this repository.

## Usage

Train the model:

```bash
python src/train.py
```

Run the app:

```bash
streamlit run app.py
```

Run the robustness check (see next section):

```bash
python src/leakage_check.py
```

## How It Works

1. **Preprocessing:** the article text is cleaned and normalised.
2. **TF-IDF:** each article becomes a weighted word vector. The weight of a term *t* in a document *d* is

   `tf-idf(t, d) = tf(t, d) × ( ln((1 + N) / (1 + df(t))) + 1 )`

   where *N* is the number of documents and *df(t)* is the number of documents containing *t* (the scikit-learn default, smooth idf).
3. **Classifier:** Logistic Regression outputs a probability between 0% and 100% that the article is fake.
4. **Explanation:** each word's contribution is its TF-IDF value times its model coefficient, so the app can show which words pushed the prediction in each direction.

## Results

The held-out evaluation in `reports/metrics.json` and the confusion matrix below are produced by `src/train.py`:

![Confusion matrix](reports/confusion_matrix.png)

Open `reports/metrics.json` for the exact accuracy, precision, recall and F1 values from your run. The evaluation set contains 1,000 articles (500 fake, 500 real).

## Known Limitations

- **The ISOT dataset has well-known shortcuts.** Most real articles begin with a source tag such as `CITY (Reuters) -`, and the dataset contains duplicate articles. A model can score very high by learning these artifacts rather than anything about whether the content is true.
- **A high score on ISOT is therefore not proof that the model detects misinformation.** It means the model separates ISOT's two sources well.
- **The model reads writing style and vocabulary, not facts.** It does not check claims against any source, so it should not be used as a fact-checker.
- **It may not generalise to other news sources, topics or time periods** without retraining and testing on data from those sources.

## Robustness Check

`src/leakage_check.py` measures how much of the score depends on those artifacts. It drops duplicate texts, then trains the same TF-IDF + Logistic Regression model on the same train/test split at three levels:

| Variant | What is removed |
|---|---|
| Baseline | nothing (duplicates dropped only) |
| Reuters tag removed | the `(Reuters)` prefix and the word "reuters" |
| Source tags removed | the above, plus "featured image", "getty images", `pic.twitter.com` links and similar publisher artifacts |

It prints accuracy, precision, recall, F1, the confusion matrix and the top words for each class, and saves everything to `reports/leakage_check_results.json`. You can also test on a different dataset:

```bash
python src/leakage_check.py --external data/other.csv --text-col text --label-col label --fake-value 1
```

If accuracy barely changes after cleaning, the model is not relying on those tags. If it drops a lot, the baseline score was inflated.

<!-- After running the check, paste your own results table here. -->

## Tech Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Matplotlib

## License

MIT, see [LICENSE](LICENSE).
