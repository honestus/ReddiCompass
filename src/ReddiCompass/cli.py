# mypack/cli.py

import argparse, warnings
import pandas as pd
from pathlib import Path
from ReddiCompass.features_extraction_and_classification.feature_extraction import extract_features
from ReddiCompass.features_extraction_and_classification.main import train_pipeline, predict_pipeline


def load_data(input_file, curr_col=None):
    """If it's a file path, load file contents; else return as-is or list of one."""
    try:
        input_file = Path(input_file).resolve()
    except:
        raise ValueError("Wrong texts' file in input. Impossible to load texts")
    if input_file.suffix ==".csv":
        return pd.read_csv(input_file)[curr_col]
    if input_file.suffix ==".parquet":
        return pd.read_parquet(input_file, columns=[curr_col])[curr_col]
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except:
        raise TypeError("Wrong texts' file in input. Impossible to load texts")
    

    texts = load_data(input_file=args.texts, curr_col='text')
    categories = load_data(input_file=args.categories, curr_col='category')

def run_extract_features():
    parser = argparse.ArgumentParser(description="Extract features from texts.")
    parser.add_argument("-f","--filename", type=str, help="Input filename for loading texts.")
    parser.add_argument("--categories", type=str, help="Optional - input filename for loading categories (only needed if extracting tf-idf). If not provided, will load from provided --filename.")
    parser.add_argument("-res","--resume-dir", type=str, help="Optional - path/directory to resume partially extracted features. If set, all the other parameters will be ignored", default=None)
    parser.add_argument("-tfidf", "--extract-tfidf", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("--saving-directory", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=-1)
    args = parser.parse_args()

    if not args.resume_dir:
        if not args.filename:
            parser.error(
                "When not resuming, --texts must be provided.\n"
                "Hint: use --resume-dir to resume from a previous run, or provide --texts to extract features."
            )

        texts = load_data(input_file=args.filename, curr_col='text')
        if args.extract_tfidf:
            categories = load_data(input_file=args.categories if args.categories else args.filename, curr_col='category')
        else:
            if args.categories:
                warnings.warn("Categories' file was provided but no categories are needed for extracting when extract-tfidf is False. They will be ignored. Use --extract-tfidf to also include tf-idf vectors with the features.")
            categories = None

    else:
        if args.filename or args.categories:
            warnings.warn("Parameters were provided in input, but they will be ignored since resume_dir was provided. All the parameters will be loaded from that resume directory")
        texts = None
        categories = None

    extract_features(
        texts=texts,
        categories=categories,
        extract_tfidf=args.extract_tfidf,
        fit_tfidf=True,
        save=args.save,
        saving_directory=args.saving_directory,
        resume_dir=args.resume_dir,
        batch_size=args.batch_size,
    )



def run_train_pipeline():
    parser = argparse.ArgumentParser(description="Train a new model.")
    parser.add_argument("-f","--filename", type=str, help="Input filename for loading texts.")
    parser.add_argument("--categories", type=str, help="Optional - input filename for loading categories. If not provided, will load from provided --filename.")
    parser.add_argument("-res","--resume-dir", type=str, help="Optional - path/directory to resume partially extracted features. If set, all the other parameters will be ignored", default=None)
    parser.add_argument("--no-tfidf", dest='extract_tfidf', help="Optional - if provided the current model wont extract tfidf vectors among its features", action="store_false")
    parser.set_defaults(extract_tfidf=True)
    parser.add_argument("--ngram-range", nargs=2, type=int, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("--saving-directory", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=-1)
    args = parser.parse_args()

    if not args.resume_dir:
        if (not args.filename):
            parser.error(
                "When not resuming, you must provide both --texts and --categories.\n"
                "Hint: use --resume-dir to resume from a previous run, or provide both data sources."
            )

        texts = load_data(input_file=args.filename, curr_col='text')
        categories = load_data(input_file=args.categories if args.categories else args.filename, curr_col='category')
    else:
        if args.filename or args.categories:
            warnings.warn("Parameters were provided in input, but they will be ignored since resume_dir was provided. All the parameters will be loaded from that resume directory")
        categories = None
        texts = None
        
    train_pipeline(
        texts=texts,
        categories=categories,
        extract_tfidf=args.extract_tfidf,
        ngram_range=tuple(args.ngram_range) if args.ngram_range else None,
        top_k=args.top_k if args.top_k else None,
        save=args.save,
        saving_directory=args.saving_directory,
        resume_dir=args.resume_dir,
        batch_size=args.batch_size,
    )




def run_predict_pipeline():
    parser = argparse.ArgumentParser(description="Run predictions using a saved pipeline.")
    parser.add_argument("-f","--filename", type=str, help="Input filename for loading texts.")
    parser.add_argument("--model", help="Path to previously trained pipeline file.")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--saving-directory", default=None)
    parser.add_argument("--batch-size", type=int, default=-1)
    args = parser.parse_args()

    if not args.resume_dir:
        if not args.filename or not args.model:
            parser.error(
                "When not resuming, both --texts and --model must be provided.\n"
                "Hint: use --resume-dir to resume from a saved pipeline, or provide both --texts and --model."
            )
    
        import joblib
        pipeline = joblib.load(args.model)
        texts = load_data(input_file=args.filename, curr_col='text')
        
    else:   
        if args.filename or args.model:
            warnings.warn("Parameters were provided in input, but they will be ignored since resume_dir was provided. All the parameters will be loaded from that resume directory")
        pipeline = None
        texts = None
        

    predict_pipeline(
        texts=texts,
        pipeline=pipeline,
        save=args.save,
        saving_directory=args.saving_directory,
        resume_dir=args.resume_dir,
        batch_size=args.batch_size,
    )

