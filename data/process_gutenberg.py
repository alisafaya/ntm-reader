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
        and len(paragraph) > 100
    )


def process_gutenberg(raw_book):
    """
    Processes a gutenberg file and returns cleaned raw text.
    """
    book = raw_book

    # Clean encoding errors
    # Replace invalid quotes with valid ones
    book = re.sub(r"â[¦]", "'", raw_book)
    book = re.sub(r"Ã©", "é", book)

    # Remove illustrations
    book = re.sub(r"\[.*(\n.*){0,10}\]", "", book)

    # italics
    book = rstrip(re.sub(r"_(.*?)_", r" \1 ", book))
    book = rstrip(re.sub(r"\[(.*?)\]", r" \1 ", book))

    # Remove >
    book = rstrip(re.sub(r"^\>", " ", book, flags=re.MULTILINE))

    # Remove license and header
    endidx = next(
        re.finditer(
            r"\n.*(\*\*\*.*END OF.*\*\*\*|End.*Project Guten.*|\*THE END.\*)",
            book,
            re.MULTILINE,
        ),
        None,
    )
    book = book[: endidx.start()] if endidx else book

    def remove_header(book):
        startidx = next(
            re.finditer(
                r"("
                "\*\*\*.*START OF.*\*\*\*"
                "|Produced by.*"
                "|This etext was produced.*"
                "|E\-text prepared by .*"
                "|.*Transcribed from.*"
                "|.*Project Gutenberg's Etext of.*"
                ")(\n.*){0,3}\n\n",
                book,
                re.MULTILINE,
            ),
            None,
        )
        if startidx and startidx.end() < 10000:
            return remove_header(book[startidx.end() :])
        return book

    book = remove_header(book)

    # Split chapters
    chapters = re.compile("\n\n\n+|\*\ *\*\ *\*\ *\*\ *\*\ *", re.MULTILINE).split(book)

    # Split chapters into paragraphs
    chapters = list(map(re.compile("\n\n", re.MULTILINE).split, chapters))

    # Fix paragraph formatting by removing word wrap.
    chapters = [
        [
            re.sub(r"\n(?=[^\n])", " ", paragraph).strip()
            for paragraph in chapter
            if paragraph.strip() and not paragraph.isupper()
        ]
        for chapter in chapters
    ]

    # Join paragraphs
    chapters = [rstrip(re.sub(r"\ +", " ", ". ".join(chapter))) for chapter in chapters]

    # Remove lists
    chapters = [
        rstrip(
            re.sub(
                r"(^((C|c)hapter|CHAPTER)*\ *[\dIVX]+.{0,200}\n+){3,}",
                "",
                chapter,
                flags=re.MULTILINE,
            )
        )
        for chapter in chapters
    ]

    # Remove redundant chapters (ratio of non alphabetics,
    # or shorter chapters maybe, contains any of the keywords)
    chapters = [chapter for chapter in chapters if filter_pargraphs(chapter)]

    # Clean double periods
    chapters = [re.sub(r"\.\ ?\.", ".", chapter) for chapter in chapters]
    chapters = [rstrip(re.sub(r"\ +", " ", chapter)) for chapter in chapters]

    book = ("\n\n").join(chapters)
    return book


def process_jsonl(jsonline):
    raw_book = json.loads(jsonline)
    return json.dumps(process_gutenberg(raw_book), ensure_ascii=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--input-jsonl",
        type=str,
        help="Path of the jsonl file containing Gutenberg books one book per line.",
    )
    argparser.add_argument(
        "--output-jsonl",
        type=str,
        help="Directory to save the output in the same format as the input.",
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes for cleaning books",
    )
    argparser.add_argument(
        "--total-books", type=int, default=None, help="Number of books to process"
    )
    argparser.add_argument(
        "--min-length",
        type=int,
        default=5000,
        help="Minimum length (chars) of a book to be considered",
    )

    args = argparser.parse_args()

    # Create the pool of workers
    pool = Pool(args.num_workers)

    # Process the books
    skipped_books, total_books = 0, 0
    with open(args.input_jsonl, "r") as fi, open(args.output_jsonl, "w") as fo:
        for book in tqdm(
            pool.imap_unordered(process_jsonl, fi, chunksize=8), total=args.total_books
        ):
            if len(book) > args.min_length:
                fo.write(book + "\n")
                total_books += 1
            else:
                skipped_books += 1

    print("Skipped {} books".format(skipped_books))
    print("Saved {} books".format(total_books))
    print("Done!")
