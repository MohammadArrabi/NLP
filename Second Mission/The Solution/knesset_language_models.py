import os
import sys
import json
import math
from collections import defaultdict, Counter

class Trigram_LM:
    def __init__(self, sentences):
        self.sentences = sentences
        self.trigrams = defaultdict(Counter)
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()
        self.num_all_tokens = 0
        self.vocabulary = set()
        self.build_the_model()

    def build_the_model(self):
        for sentence in self.sentences:
            tokens = ['<s_0>', '<s_1>'] + sentence.split()
            self.num_all_tokens += len(tokens)
            for token in tokens:
                self.vocabulary.add(token)
            for i in range(len(tokens)):
                self.unigrams[tokens[i]] += 1
                if i < len(tokens) - 1:
                    self.bigrams[(tokens[i])][tokens[i + 1]] += 1
                if i < len(tokens) - 2:
                    self.trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

    def calculate_prob_of_sentence(self, sentence, smoothing_type):
        tokens = ['<s_0>', '<s_1>'] + sentence.split()
        vocabulary_size = len(self.vocabulary)
        corpus_size = self.num_all_tokens
        log_propapility = 0

        if smoothing_type == "Laplace":
            for i in range(len(tokens) - 2):
                bigrams_count = self.bigrams[(tokens[i])][tokens[i + 1]]
                trigrams_count = self.trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]]
                propapility = (trigrams_count + 1) / (bigrams_count + vocabulary_size)
                log_propapility += math.log(propapility)

        elif smoothing_type == "Linear":
            l1, l2, l3 = 0.7, 0.2, 0.1
            for i in range(len(tokens) - 2):
                trigrams_count = self.trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]]
                bigrams_count2 = self.bigrams[(tokens[i])][tokens[i + 1]]
                bigrams_count = self.bigrams[(tokens[i + 1])][tokens[i + 2]]
                unigrams_count = self.unigrams[tokens[i + 2]]
                unigrams_count2 = self.unigrams[tokens[i + 1]]
                if(bigrams_count2 != 0):
                     propapility_trigrams = (trigrams_count + 1) / (bigrams_count2)
                else:
                     propapility_trigrams = 0
                if(unigrams_count2 != 0):
                     propapility_bigrams = (bigrams_count + 1) / (unigrams_count2)
                else:
                    propapility_bigrams = 0

                propapility_unigram = (unigrams_count + 1) / (corpus_size + vocabulary_size)

                p = l1 * propapility_trigrams + l2 * propapility_bigrams + l3 * propapility_unigram
                log_propapility += math.log(p)

        return log_propapility

    def generate_next_token(self, sentence):
        best_p = -math.inf
        for token in self.vocabulary:
            sen = sentence + " " + token
            p_next_token = self.calculate_prob_of_sentence(sen,"Linear")
            if(p_next_token > best_p and token != "<s_0>" and token != "<s_1>"):
                best_p = p_next_token
                next_token = token
        return next_token

    def get_k_n_collocations(self, k, n, corpus, measure_type):
     collocations = defaultdict(int)
     for sentence in corpus:
         tokens = sentence.split()
         if len(tokens) < 2:
             tokens = ['<s_0>', '<s_1>'] + tokens
         for i in range(len(tokens) - n + 1):
             collocation = ' '.join(tokens[i:i + n])
             collocations[collocation] += 1

     if measure_type == "frequency":
         sort_collocations = sorted(collocations.items(), key=lambda item: item[1], reverse=True)

     elif measure_type == "tfidf":
         tfidf_list = defaultdict(list)
         corpus_size = len(corpus)
         collocation_count = defaultdict(int)
         for sentence in corpus:
             tokens = sentence.split()
             if len(tokens) < 2:
                 tokens = ['<s_0>', '<s_1>'] + tokens
             unique_collocations = set()
             for i in range(len(tokens) - n + 1):
                 collocation = ' '.join(tokens[i:i + n])
                 unique_collocations.add(collocation)
             for collocation in unique_collocations:
                 collocation_count[collocation] += 1

         for sentence in corpus:
             tokens = sentence.split()
             if len(tokens) < 2:
                 tokens = ['<s_0>', '<s_1>'] + tokens
             num_tokens_in_sentence = len(tokens)
             collocation_frequency = defaultdict(int)
             for i in range(len(tokens) - n + 1):
                 collocation = ' '.join(tokens[i:i + n])
                 collocation_frequency[collocation] += 1

             for collocation, frequency in collocation_frequency.items():
                 tf = frequency / num_tokens_in_sentence
                 idf = math.log(corpus_size / (1 + collocation_count[collocation]))
                 tfidf = tf * idf
                 tfidf_list[collocation].append(tfidf)

         avg_tfidf_list = {}
         for collocation, count in tfidf_list.items():
             avg = sum(count) / len(count)
             avg_tfidf_list[collocation] = avg
         sort_collocations = sorted(avg_tfidf_list.items(), key=lambda item: item[1], reverse=True)

     common_k_collocations = []
     for collocation, _ in sort_collocations:
         if not (") )" in collocation):
             common_k_collocations.append(collocation)

     return common_k_collocations[:k]

def write_collocations_file(output_path, file_name, committee_model, plenary_model):
    file_path = os.path.join(output_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        for n in [2, 3, 4]:
            for measure in ["frequency", "tfidf"]:
                if(n == 2):
                    num = "Two"
                elif(n == 3):
                    num = "Three"
                elif(n == 4):
                    num = "Four"
                if measure == "frequency":
                   f.write(f"{num}-gram collocations:\n")
                   f.write(f"Frequency:\n")
                elif measure == "tfidf":
                   f.write(f"TF-IDF:\n")
                f.write("Committee corpus:\n")
                committee_collocations = committee_model.get_k_n_collocations(10, n, committee_model.sentences, measure)
                for collocation in committee_collocations:
                    f.write(collocation + "\n")
                f.write("\n")

                f.write("Plenary corpus:\n")
                plenary_collocations = plenary_model.get_k_n_collocations(10, n, plenary_model.sentences, measure)
                for collocation in plenary_collocations:
                    f.write(collocation + "\n")
                f.write("\n")

def complete_masked_sentences(model, sentences):
    completed_sentences_list = []
    ret_generated_tokens_list = []
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) < 2:
            tokens = ['<s_0>', '<s_1>'] + tokens
        completed_sentence = ""
        generated_tokens_list = []
        for token in tokens:
            if token == '[*]':
                generated_token = model.generate_next_token(completed_sentence)
                generated_tokens_list.append(generated_token)
                completed_sentence += ' ' + generated_token
            else:
                completed_sentence += ' ' + token

        completed_sentence = completed_sentence.strip()
        completed_sentences_list.append(completed_sentence)
        ret_generated_tokens_list.append(generated_tokens_list)
    return completed_sentences_list , ret_generated_tokens_list

def committee_or_plenary(committee_probability, plenary_probability):
    corpus = []
    i=0
    for probability_committee in committee_probability:
        probability_plenary = plenary_probability[i]
        i+=1
        if probability_committee > probability_plenary:
            corpus.append("committee")
        else:
            corpus.append("plenary")
    return corpus


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('must be 3 args')
        sys.exit(1)
    jsonl_file_path = sys.argv[1]
    masked_sentences_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    protocols = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                protocols.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i}: {e}")

    committee_sentences = []
    plenary_sentences = []
    for protocol in protocols:
        if(protocol['protocol_type'] == 'committee'):
            committee_sentences.append(protocol['sentence_text'])
        else:
            plenary_sentences.append(protocol['sentence_text'])

    committee_model = Trigram_LM(committee_sentences)
    plenary_model = Trigram_LM(plenary_sentences)

    write_collocations_file(output_file_path , 'knesset_collocations.txt', committee_model, plenary_model)

    masked_sentences = []

    with open(masked_sentences_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                masked_sentences.append(stripped_line)

    completed_sentences_committee , generated_tokens_committee = complete_masked_sentences(committee_model, masked_sentences)
    completed_sentences_plenary , generated_tokens_plenary = complete_masked_sentences(plenary_model, masked_sentences)

    probability_committee_committee = []
    for sentence in completed_sentences_committee:
        probability = committee_model.calculate_prob_of_sentence(sentence, "Linear")
        probability_committee_committee.append(probability)
    probability_committee_plenary = []
    for sentence in completed_sentences_plenary:
        probability = committee_model.calculate_prob_of_sentence(sentence, "Linear")
        probability_committee_plenary.append(probability)
    probability_plenary_plenary = []
    for sentence in completed_sentences_plenary:
        probability = plenary_model.calculate_prob_of_sentence(sentence, "Linear")
        probability_plenary_plenary.append(probability)
    probability_plenary_committee = []
    for sentence in completed_sentences_committee:
        probability = plenary_model.calculate_prob_of_sentence(sentence, "Linear")
        probability_plenary_committee.append(probability)

    corpus_committee = committee_or_plenary(probability_committee_committee, probability_plenary_committee)
    corpus_plenary = committee_or_plenary(probability_committee_plenary, probability_plenary_plenary)

    file_path = os.path.join(output_file_path, 'sentences_results.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, masked_sentence in enumerate(masked_sentences):

            masked_sentence = masked_sentence.strip()
            f.write(f"Original sentence: {masked_sentence}\n")
            f.write(f"Committee sentence: {completed_sentences_committee[i]}\n")
            f.write(f"Committee tokens: {','.join(generated_tokens_committee[i])}\n")
            f.write(f"Probability of committee sentence in committee corpus: {probability_committee_committee[i]:.3f}\n")
            f.write(f"Probability of committee sentence in plenary corpus: {probability_plenary_committee[i]:.3f}\n")
            f.write(f"This sentence is more likely to appear in corpus: {corpus_committee[i]}\n")
            f.write(f"Plenary sentence: {completed_sentences_plenary[i]}\n")
            f.write(f"Plenary tokens: {','.join(generated_tokens_plenary[i])}\n")
            f.write(f"Probability of plenary sentence in plenary corpus: {probability_plenary_plenary[i]:.3f}\n")
            f.write(f"Probability of plenary sentence in committee corpus: {probability_committee_plenary[i]:.3f}\n")
            f.write(f"This sentence is more likely to appear in corpus: {corpus_plenary[i]}\n")
            f.write("\n")