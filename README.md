# The ?th place solution of Hungry Geese Competition in Kaggle.

This is an implementation of the 5th place solution of Hungry Geese Competition in Kaggle.
The details of the solution is described in Kaggle Discussion.

Main programs are in following files:
- `src/agent.py`: program of goose agents
- `src/model.py`: program of CNN models

## How to use

### Generate game records
Following command generates game records and saves it in the data directory.
```
python src/generate.py [--workers <num of threads>]
```

### Train a model
Following command trains a new model and saves it in the data directory.
```
python src/train.py [--workers <num of threads>]
```

### Train a distilled model
Following command trains a distilled model and saves it in the data directory.
```
python src/distill.py [--workers <num of threads>]
```

### Choose the strongest model
Following command evaluates the latest models and saves the strongest model at `data/99999_model.pkl`.
```
python src/evaluate.py [--workers <num of threads>]
```

### Make a submission file
Following command makes a submission file at `data/submission.py`.
```
python src/make.py <model file (*.pkl)>
```

