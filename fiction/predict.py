import os
import json
import random
import pickle

from xgboost import XGBClassifier

from tqdm import tqdm

from .preprocess import *
from .train import get_bow_vector


def predict(infile, outfile, tokenizer):

    xgb_model = pickle.load(open("xgb_clf_poetry.pkl", "rb"))
    with open(outfile, "w") as fo:
        with open(infile) as fi:
            for idx, l in tqdm(enumerate(fi)):
                ids = np.array(tokenizer.encode(json.loads(l)).ids)
                if xgb_model.predict(get_bow_vector(ids)[np.newaxis, :])[0] == 1:
                    fo.write(l)
                    print(idx)

    
if __name__ == "__main__":
    import sys
    
    infile, outfile = sys.argv[1], sys.argv[2]
    
    tokenizer = Tokenizer.from_file("data/bookcorpus1_word_tokenizer.json")

    tokenizer.model.unk_token = '<unk>'
    tokenizer.enable_padding(direction='left', pad_id=0, pad_type_id=0, pad_token='<pad>', length=2**16)
    tokenizer.enable_truncation(2**16)
    
    predict(infile, outfile, tokenizer)
