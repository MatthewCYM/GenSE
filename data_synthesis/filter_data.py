import argparse
import os
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    with open(os.path.join(args.output_dir, 'classification_predictions.txt'), 'r', encoding='utf-8') as f:
        predicted_results = [line.strip() for line in f.readlines()]
    with open(os.path.join(args.output_dir, 'classification_logits.txt'), 'r', encoding='utf-8') as f:
        predicted_logits = [line.strip().split('\t') for line in f.readlines()]

    # predicted_results = predicted_results[1:]
    print(len(predicted_results))
    print(len(predicted_logits))

    all_data = pd.read_csv(os.path.join(args.output_dir, 'test.csv'))
    sents0, sents1, sents2 = all_data['sent0'], all_data['sent1'], all_data['hard_neg']
    print(len(sents0))

    filtered_data = []
    for i in tqdm(range(0, len(predicted_results), 2)):
        original_sent = sents0[i // 2]
        positive_sent = sents1[i // 2]
        negative_sent = sents2[i // 2]

        if predicted_results[i] != 'entailment':
            continue
        if predicted_results[i+1] != 'contradiction':
            continue

        # based on the confidence
        positive_logits = predicted_logits[i]
        negative_logits = predicted_logits[i+1]
        positive_logits = [float(item) for item in positive_logits]
        positive_logits = softmax(positive_logits)
        if positive_logits[0] < 0.9:
            continue
        negative_logits = [float(item) for item in negative_logits]
        negative_logits = softmax(negative_logits)
        if negative_logits[1] < 0.9:
            continue

        original_sent = str(original_sent)
        positive_sent = str(positive_sent)
        negative_sent = str(negative_sent)
        if len(original_sent.split()) < 8:
            continue
        if len(positive_sent.split()) < 5:
            continue
        if len(negative_sent.split()) < 5:
            continue

        # based on the edit distance
        if positive_sent[:-1] in original_sent or original_sent[:-1] in positive_sent:
            continue
        if negative_sent[:-1] in original_sent or original_sent[:-1] in negative_sent:
            continue

        tmp = [original_sent, positive_sent, negative_sent]
        filtered_data.append(tmp)
    print(len(filtered_data))
    filtered_data = pd.DataFrame(filtered_data)

    filtered_data.to_csv(os.path.join(args.output_dir, 'synli.csv'),
                         header=['sent0',
                                 'sent1',
                                 'hard_neg'],
                         index=False)


if __name__ == '__main__':
    main()