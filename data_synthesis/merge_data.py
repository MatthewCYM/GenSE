import os
import argparse
import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    sents0 = load_dataset('text', data_files=args.input_file, cache_dir='cache')
    sents0 = sents0['train']['text']


    with open(os.path.join(args.output_dir, 'generated_predictions.txt'), 'r', encoding='utf-8') as f:
        sents12 = [line.strip() for line in f.readlines()]
    sents1 = [sent for idx, sent in enumerate(sents12) if idx % 2 == 0]
    sents2 = [sent for idx, sent in enumerate(sents12) if idx % 2 == 1]
    assert len(sents0) == len(sents1) == len(sents2)
    print(len(sents0))
    sents = []
    for idx in range(len(sents0)):
        if sents1[idx].strip() == '':
            sents1[idx] = " "
        if sents2[idx].strip() == '':
            sents2[idx] = " "
        sents.append([sents0[idx], sents1[idx], sents2[idx]])

    sents = pd.DataFrame(sents)
    sents.to_csv(os.path.join(args.output_dir, 'test.csv'), header=['sent0', 'sent1', 'hard_neg'], index=False)


if __name__ == '__main__':
    main()
