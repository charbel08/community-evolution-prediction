## Install Dependencies
```
pip install -r requirements.txt
```

## How to Run
Make sure you are in the root directory.
### Data Preprocessing
```
python -m scripts.preprocess --raw_data_file soc-redditHyperlinks-body.tsv --dataset reddit --granularity M
```

### Pipeline
```
python -m src.main --config config/louvain_twitter.yaml
```