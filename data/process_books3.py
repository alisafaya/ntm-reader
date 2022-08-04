import re
import json
import argparse

from multiprocessing import Pool
from tqdm import tqdm

KEYWORDS = [
    "Transcriber's Note",
    "CONTENTS",
    "Copyright",
    "copyright",
    "ISBN",
    "EBOOK",
    "©",
]


def rstrip(line):
    return re.sub(r"^ +", "", line, flags=re.MULTILINE)


def filter_pargraphs(paragraph):
    return (
        len(re.findall(r"[a-zA-Z]", paragraph)) > (len(paragraph) / 3)
        and not any(keyword in paragraph for keyword in KEYWORDS)
        and len(paragraph) > 2
    )


def process_books3(raw_book):
    """
    Processes a books3 file and returns cleaned raw text.
    """
    book = raw_book

    # Remove lists
    book = rstrip(
        re.sub(
            r"(^(?:\*\*).{1,100}(?:\*\*).{0,100}\n+){2,}", "", book, flags=re.MULTILINE
        )
    )
    book = rstrip(re.sub(r"(^\d.{0,100}\n+){3,}", "", book, flags=re.MULTILINE))
    book = rstrip(re.sub(r"(^.{0,20}\n+){10,}", "\n", book, flags=re.MULTILINE))

    # Clean encoding errors
    # Replace invalid quotes with valid ones
    book = rstrip(re.sub(r"â[¦]", "'", book))
    book = rstrip(re.sub(r"Ã©", "é", book))

    # italics
    book = rstrip(re.sub(r"_(.*?)_", r" \1 ", book))
    book = rstrip(re.sub(r"\[(.*?)\]", r" \1 ", book))

    # Remove illustrations
    book = rstrip(re.sub(r"^\[.*(\n.*){0,10}\]", " ", book, flags=re.MULTILINE))

    # Remove >
    book = rstrip(re.sub(r"^\>", " ", book, flags=re.MULTILINE))

    # Remove Headers # ## ### ...
    book = rstrip(re.sub(r"^(#){1,6}.{1,100}$", "", book, flags=re.MULTILINE))
    book = rstrip(
        re.sub(r"^(\*\*).{1,100}(\*\*).{0,100}$", "", book, flags=re.MULTILINE)
    )
    book = rstrip(
        re.sub(r"^((C|c)hapter|CHAPTER).{1,100}$", "", book, flags=re.MULTILINE)
    )

    # Remove ** Bold ** markers
    book = rstrip(re.sub(r"(?:\*\*)", " ", book))

    # Split into chapters and paragraphs
    chapters = re.compile(r"\n\n\n+", re.MULTILINE).split(book)
    chapters = [
        re.compile("\n\n", re.MULTILINE).split(chapter)
        for chapter in chapters
        if len(chapter) > 100
    ]

    # Remove redundant chapters (ratio of non alphabetics,
    # or shorter chapters maybe, contains any of the keywords)
    chapters = [list(filter(filter_pargraphs, chapter)) for chapter in chapters]

    # Join paragraphs
    chapters = [rstrip(re.sub(r"\ +", " ", ". ".join(chapter))) for chapter in chapters]

    # Clean double periods
    chapters = [re.sub(r"\.\ ?\.", ".", chapter) for chapter in chapters]
    chapters = [rstrip(re.sub(r"\ +", " ", chapter)) for chapter in chapters]

    book = ("\n\n").join(chapters)
    return book


def process_jsonl(jsonline):
    raw_book = json.loads(jsonline)
    return json.dumps(process_books3(raw_book), ensure_ascii=False)


import sys
import os
from glob import glob

if __name__ == "__main__":
    inputdir, outputdir = sys.argv[1:]
    inputfiles = glob(os.path.join(inputdir, "*.txt"))
    outputfiles = [
        os.path.join(outputdir, os.path.basename(inputfile)) for inputfile in inputfiles
    ]

    for i, o in zip(inputfiles, outputfiles):
        print(i, o)
        with open(i, "r") as f:
            raw_book = f.read()
        with open(o, "w") as f:
            f.write(process_books3(raw_book))


# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser()

#     argparser.add_argument(
#         "--input-jsonl",
#         type=str,
#         help="Path of the jsonl file containing Books3 book corpus, one book per line.",
#     )
#     argparser.add_argument(
#         "--output-jsonl",
#         type=str,
#         help="Directory to save the output in the same format as the input.",
#     )
#     argparser.add_argument(
#         "--num-workers",
#         type=int,
#         default=8,
#         help="Number of processes for cleaning books",
#     )
#     argparser.add_argument(
#         "--total-books", type=int, default=None, help="Number of books to process"
#     )
#     argparser.add_argument(
#         "--min-length",
#         type=int,
#         default=5000,
#         help="Minimum length (chars) of a book to be considered",
#     )
#     args = argparser.parse_args()

#     # Create the pool of workers
#     pool = Pool(args.num_workers)

#     # Process the books
#     skipped_books, total_books = 0, 0
#     with open(args.input_jsonl, "r") as fi, open(args.output_jsonl, "w") as fo:
#         for book in tqdm(
#             pool.imap_unordered(process_books3, fi, chunksize=8), total=args.total_books
#         ):
#             if len(book) > args.min_length:
#                 fo.write(book + "\n END OF BOOK \n")
#                 total_books += 1
#             else:
#                 skipped_books += 1

#     print("Skipped {} books".format(skipped_books))
#     print("Saved {} books".format(total_books))
#     print("Done!")
