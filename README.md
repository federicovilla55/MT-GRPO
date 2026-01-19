# DeepLearningProject

![Model Pipeline](/assets/main-figure.png)

## Data Download

To get started with the project, first download the data from the following links:

- [WMT25 General MT Humeval Data](https://github.com/wmt-conference/wmt25-general-mt/blob/main/data/wmt25-genmt-humeval.jsonl)
- [Tatoeba English Sentences](https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2)

After downloading, create a folder named `data` (if it does not already exist) in the project root and place both files inside this `data` folder.

Make sure the paths look like this:

- `data/wmt25-genmt-humeval.jsonl`
- `data/eng_sentences.tsv`

## Python Environment

Create a Python virtual environment:

```bash
python3 -m venv deep_learning
source deep_learning/bin/activate
pip install -r requirements.txt
```

## Model Checkpoints

Model checkpoints are available in a [public polybox folder](https://polybox.ethz.ch/index.php/s/Zx7YNW2RPoMKJtk):

- [llama-8b-sentinel](https://polybox.ethz.ch/index.php/s/syzEiYDJtt7Pxyy)
- [qwen-4b-sentinel](https://polybox.ethz.ch/index.php/s/eFioWRGB8oW4fkj)