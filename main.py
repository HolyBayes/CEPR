import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
import argparse
from tqdm.auto import tqdm
from collections import Counter
import json


CKPT_PATH = 'checkpoints/CEPR_model_xgb.pkl'

parser = argparse.ArgumentParser(description='Script for spectrum classification')
parser.add_argument('input_file', type=str)
parser.add_argument('--n-samples', type=int, default=1000, help='Number of subsamples of file. The more the more accurate the prediction is')
parser.add_argument('--sample-size', type=int, default=1000, help='Subsample size. The more the more accurate the prediction is')
args = parser.parse_args()

with open(CKPT_PATH, 'rb') as f:
  clf = pickle.load(f)


def predict(X, args):
  return clf.predict(X)

def prepare_dataset(df, args):
  feature_fns = []
  def feature_fn(vals, phase, percentile):
    return np.percentile(vals[:,phase], percentile)

  for phase in range(3):
    for percentile in range(0,100,10):
      fn_args = (phase,percentile)
      feature_fns.append((feature_fn, fn_args))

  X = []
  file_values = df.values
  for _ in tqdm(range(args.n_samples)):
    sample_indices = np.random.choice(np.arange(file_values.shape[0]), args.sample_size)
    samples = file_values[sample_indices]
    x = []
    for fn, fn_args in feature_fns:
      x.append(fn(samples, *fn_args))
    X.append(x)
  X = np.array(X)
  return X

def main(args):
  df = pd.read_csv(args.input_file)
  X = prepare_dataset(df, args)
  preds = predict(X, args)
  stat = Counter(preds)
  preds = {{0: 'A', 1: 'B', 2: 'C', 3: 'D'}[k]:v/X.shape[0] for k,v in stat.items()}
  print(json.dumps(preds))

if __name__ == '__main__':
  print(args)
  main(args)