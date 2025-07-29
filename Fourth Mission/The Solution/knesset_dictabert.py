import sys
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('must be 2 args')
        sys.exit(1)
    masked_sentences_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    masked_sentences = []
    with open(masked_sentences_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                masked_sentences.append(stripped_line)

    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
    model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')

    num_of_sentence = 0
    DictaBERT_tokens_list = []
    DictaBERT_sentences_list = []
    for sentence in masked_sentences:
        num_of_stars = sentence.count('[*]')
        sentence = sentence.replace('[*]', '[MASK]')
        current_tokens_list = []
        for i in range(num_of_stars):
            index = sentence.split(' ').index('[MASK]')
            output = model(tokenizer.encode(sentence, return_tensors='pt'))
            top_1 = torch.topk(output.logits[0, index+1, :],1)[1]
            dict_token = tokenizer.convert_ids_to_tokens([top_1])[0]
            current_tokens_list.append(dict_token)
            sentence = sentence.replace('[MASK]', dict_token,1)

        DictaBERT_tokens_list.append(current_tokens_list)
        DictaBERT_sentences_list.append(sentence)

    output_file_path = os.path.join(output_file_path, 'dictabert_results.txt')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i, masked_sentence in enumerate(masked_sentences):
            masked_sentence = masked_sentence.strip()
            f.write(f"Original sentence: {masked_sentence}\n")
            f.write(f"DictaBERT sentence: {DictaBERT_sentences_list[i]}\n")
            f.write(f"DictaBERT tokens: {','.join(DictaBERT_tokens_list[i])}\n")
            f.write("\n")