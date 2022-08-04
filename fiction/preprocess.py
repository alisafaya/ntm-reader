import json

import numpy as np
from tqdm import trange

from tokenizers import normalizers, pre_tokenizers, Tokenizer, models, trainers

def load_data(metadata_file):

    metadata = np.array(json.load(open(metadata_file)))
    get_label = np.vectorize(lambda x: 1 if 'Poetry' in x['categories'] else 0)
    # get_label = np.vectorize(lambda x: 1 if 'Fiction' in x['categories'] else (0 if 'Nonfiction' in x['categories'] else 2))
    labels = get_label(metadata)

    return metadata[labels == 0], metadata[labels == 1], metadata[labels == 2]


def build_tokenizer(files, vocab_size=2**16, min_freq=2**3):

    tokenizer = Tokenizer(models.WordLevel())
    tokenizer.normalizer = normalizers.BertNormalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=['<pad>', '<unk>']
    )

    print("--"*20)
    print("Build word tokenizer with ", len(files), "files.")

    tokenizer.train(files, trainer)

    tokenizer.model.unk_token = '<unk>'
    tokenizer.enable_padding(direction='left', pad_id=0, pad_type_id=0, pad_token='<pad>', length=2**16)
    tokenizer.enable_truncation(2**16)

    return tokenizer


def read_chunked(files, chunk_size):

    for i in trange(0, len(files), chunk_size):
        yield [ open(os.path.join(books_path, f['book_name'])).read()[:2**22] for f in files[i:i+chunk_size] ], files[i:i+chunk_size]


def tokenize_and_save(tokenizer, bookset, outfile):
    
    with open(outfile, "w") as fo:
        for contents, books in read_chunked(bookset, 64):
            tokenized_batch = tokenizer.encode_batch(contents)

            for book, tokenized in zip(books, tokenized_batch):
                book["tokens"] = tokenized.ids
                fo.write(json.dumps(book, ensure_ascii=False) + "\n")


if __name__ == '__main__':

    import sys
    import os
    from glob import glob

    book_genres_file = sys.argv[1]
    books_path = sys.argv[2]

    all_files = glob(os.path.join(books_path, "*.txt"))

    # tokenizer = build_tokenizer(all_files)
    # tokenizer.save("data/bookcorpus1_word_tokenizer.json", pretty=True)

    tokenizer = Tokenizer.from_file("data/bookcorpus1_word_tokenizer.json")

    nonfiction, fiction, unknown = load_data(book_genres_file)

    print("Nonfiction books: ", nonfiction.shape[0])
    print("Fiction books: ", fiction.shape[0])
    print("Unknown books: ", unknown.shape[0])

    tokenize_and_save(tokenizer, nonfiction[:3000], "data/nonfiction_books_tokenized.json")
    tokenize_and_save(tokenizer, fiction, "data/fiction_books_tokenized.json")