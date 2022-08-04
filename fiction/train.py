import os
import json
import random
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from tqdm import tqdm

from .preprocess import *

def get_bow_vector(tokens, vocab_size=2**16, max_value=5000):
    v = np.zeros(vocab_size)
    unique, frequency = np.unique(tokens, return_counts = True)
    
    # trim extreme values
    frequency[ frequency > max_value ] = max_value
    v[unique] = frequency

    # l2 normalization
    v = v / np.sqrt(np.power(v, 2).sum())

    return v

def build_bow(bookset):
    bow = np.stack([ get_bow_vector(np.array(x['tokens'])) for x in tqdm(bookset, desc="Building BOW") ])
    return bow

def evaluate_baseline(fiction, nonfiction):

    print("Reading data files")

    fiction_set = [ json.loads(l) for l in open(fiction) ]
    nonfiction_set = [ json.loads(l) for l in open(nonfiction) ]

    print("Construct BOW") 

    X = build_bow(fiction_set + nonfiction_set)
    y = np.concatenate([np.ones(len(fiction_set), dtype=np.int32), np.zeros(len(nonfiction_set), dtype=np.int32)])

    del fiction_set
    del nonfiction_set

    print("X", X.shape)
    print("y", y.shape)

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    xgb_clf = XGBClassifier(random_state = 123, use_label_encoder=False)

    # Training
    print('Training on', len(X), 'samples')
    xgb_clf.fit(X, y, eval_metric='logloss', verbose=True)
    predictions = xgb_clf.predict(X)
    print ('-'* 40, '\nTraining results\n', classification_report(y, predictions, digits=3))

    # Testing
    predictions = xgb_clf.predict(X_test)
    print ('Test results\n', classification_report(y_test, predictions, digits=3))

    # save
    pickle.dump(xgb_clf, open("xgb_clf_poetry.pkl", "wb"))


if __name__ == "__main__":
    import sys
    
    fiction, nonfiction = sys.argv[1], sys.argv[2]
    evaluate_baseline(fiction, nonfiction)
