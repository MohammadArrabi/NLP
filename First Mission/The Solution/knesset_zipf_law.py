
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter


def get_key(dic, value):
    try:
        list_key = []
        for key, value1 in dic.items():
            if value == np.log2(value1):
                list_key.append(key[::-1])
        return list_key
    except Exception as e:
        print(f'Exception in get_key: {e}')


def check_word(word):
    try:
        if word == '':
            return False
        hebrew_letters = 'אבגדהוזחטיכלמנסעפצקרשת'
        return all(letter in hebrew_letters for letter in word)
    except Exception as e:
        print(f'Exception in check_word: {e}')


def clean_token(token):
    token = re.sub(r'[^א-ת]', '', token)
    return token


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {line}")
    return data


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print('must have 2 arguments')
            sys.exit(1)

        jsonl_file_name = sys.argv[1]
        output_path = sys.argv[2]

        # Read data from JSONL file
        data = read_jsonl(jsonl_file_name)

        # count frequencies
        frequncy_dictionary = Counter()
        for item in data:
            sentence = item['sentence_text']
            tokens = re.findall(r'\b\w+\b', sentence)
            for word in tokens:
                clean_word = clean_token(word)
                if check_word(clean_word):
                    frequncy_dictionary[clean_word] += 1

        sorted_tokens = sorted(frequncy_dictionary.items(), key=lambda item: item[1], reverse=True)
        ranks = [np.log2(i + 1) for i in range(len(sorted_tokens))]
        frequencies = [np.log2(count) for _, count in sorted_tokens]

        # Find most and least frequent words
        most_frequent = [word for word, _ in sorted_tokens[:10]]
        least_frequent = [word for word, _ in sorted_tokens[-10:]]

        print(f'most common words: {most_frequent}')
        print(f'least common words: {least_frequent}')

        # Plot Zipf's Law
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, frequencies, marker='o', linestyle='none')
        plt.xlabel('Log2 Rank')
        plt.ylabel('Log2 Frequency')
        plt.title("Zipf's Law")
        plt.grid(True)
        # plt.savefig(output_path)
    except Exception as e:
        print(f'Exception in main: {e}')

