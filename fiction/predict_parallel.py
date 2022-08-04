import os
import json
import itertools
import sys
import pickle
import multiprocessing as mp
import numpy as np

from xgboost import XGBClassifier
from tokenizers import Tokenizer
from tqdm import tqdm

from .train import get_bow_vector

def predict_lines(lines):

    pid = os.getpid()
    tokenizers[pid] = tokenizers.get(pid, Tokenizer.from_file("data/bookcorpus1_word_tokenizer.json"))
    books = []

    for l in lines:
        if l is not None:
            books.append(json.loads(l)[:2**19])

    batch = np.array([ sum(1 for t in x.ids if t == 1) / len(x) for x in tokenizers[pid].encode_batch(books) ])
    return batch, np.array(lines[:len(books)], dtype=object)


if __name__ == "__main__":
    
    indir = sys.argv[1]
    outdir = sys.argv[2]
    nworkers = int(sys.argv[3] if len(sys.argv) > 3 else 32)
    
    batchsize = 128
    xgb_model = pickle.load(open("xgb_clf_fiction.pkl", "rb"))
    tokenizers = {}

    with open(outdir, "w") as fo:
        with open(indir) as fi:
            with mp.Manager() as manager:
                with mp.Pool(processes=nworkers) as pool:
                    processors_iter = pool.imap(predict_lines, tqdm(itertools.zip_longest(*[fi]*batchsize)))
                    for bow, books in processors_iter:
                        fo.write("".join(books[bow < 0.1]))
