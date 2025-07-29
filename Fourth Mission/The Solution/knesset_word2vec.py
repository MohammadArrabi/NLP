import json
import sys
import numpy as np
import random
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


def embeddings_of_sentences(sentences, model):
    embeddings = []
    for sentence in sentences:
        sentence_vector = np.zeros(model.vector_size)
        sum = 0
        for word in sentence.split():
            if word in model.wv:
                sentence_vector += model.wv[word]
                sum += 1
        if sum != 0:
            sentence_vector /= sum
        embeddings.append(sentence_vector)
    return np.array(embeddings)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('must be 2 args')
        sys.exit(1)
    jsonl_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # חלק א'
    protocols = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                protocols.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i}: {e}")

    sentences = []
    for protocol in protocols:
        sentences.append(protocol['sentence_text'])

    tokenized_sentences = []
    for sentence in sentences:
        cleaned_tokens = []
        tokens = sentence.split()
        for token in tokens:
            if 1424 <= ord(token[0]) <= 1514:
               cleaned_tokens.append(token)
        tokenized_sentences.append(cleaned_tokens)

    model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1)
    model.save("knesset_word2vec.model")
    word_vectors = model.wv

    # חלק ב' סעיף א
    words_list = ['ישראל', 'כנסת', 'ממשלה', 'חבר', 'שלום', 'שולחן', 'מותר', 'מדבר', 'ועדה']
    similarity_results = {}
    corpus_words = list(word_vectors.index_to_key)
    for word1 in words_list:
        similarity = []
        for word2 in corpus_words:
            if word1 != word2:
                similarity_score = word_vectors.similarity(word1, word2)
                similarity.append((word2, similarity_score))
        sorted_similarities = sorted(similarity, key=lambda x: x[1], reverse=True)[:5]
        similarity_results[word1] = sorted_similarities

    file_path = os.path.join(output_file_path, 'knesset_similar_words.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, similar_words in similarity_results.items():
            f.write(f"{word}: ")
            f.write(', '.join([f"({sim_word}, {sim_score})" for sim_word, sim_score in similar_words]))
            f.write('\n')

    # חלק ב' סעיף ב
    sentence_embeddings = []
    for sentence in tokenized_sentences:
        sentence_vector = np.zeros(50)
        sum = 0
        for word in sentence:
            if word in word_vectors:
                sentence_vector += word_vectors[word]
                sum += 1
        if sum != 0:
            sentence_vector /= sum
        sentence_embeddings.append(sentence_vector)

    sentence_embeddings = np.array(sentence_embeddings)
    filtered_sentences = [sentence for sentence in tokenized_sentences if len(sentence) >= 4]

    # חלק ב' סעיף ג
    hebrew_sentences = [
        "אבל זה אותו דבר .",
        "ולכן , צריך להביא את זה בחשבון .",
        "אני לא מוכן לקבל את זה .",
        "אם כן , רבותי, אנחנו עוברים להצבעה .",
        "זה לא דבר שהוא חדש .",
        "מה התפקיד שלכם בנושא הזה ?",
        "בגלל שאני אומר את האמת ?",
        "בכל מקרה ההצבעה לא תתקיים היום .",
        "אני לא כל כך מבין .",
        "איך ייתכן דבר כזה ?"
    ]

    text = ""
    our_index_embeddings = embeddings_of_sentences(hebrew_sentences, model)
    # Compute cosine similarity between the provided sentences and the dataset sentences
    matrix = cosine_similarity(our_index_embeddings, sentence_embeddings)
    # Iterate over the provided sentences and find the most similar sentence in the dataset
    for i, index in enumerate(our_index_embeddings):
        text += hebrew_sentences[i] + ': most similar sentence: '
        max_index = matrix[i].argsort()[-2]  # Find the second high similarity index
        text += sentences[max_index] + '\n'  # Get the most similar sentence from the dataset

    # Remove the newline character
    text = text[:-1]

    file_path = os.path.join(output_file_path, 'knesset_similar_sentences.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    '''
    # print(f":שנאה, אהבה{model.wv.similarity('אהבה', 'שנאה')}")
    # print(f":קר, חם{model.wv.similarity('חם', 'קר')}")
    # print(f":אחרון, ראשוןן{model.wv.similarity('ראשון', 'אחרון')}")
    # print(f":עצוב, שמח{model.wv.similarity('שמח', 'עצוב')}")
    # print(f"קטן, גדול {model.wv.similarity('גדול', 'קטן')}")
    '''

    # חלק ב' סעיף ד
    sentences_with_red_words = [
         "ברוכים הבאים , הכנסו בבקשה [לדיון] .",
         "בתור יושבת ראש הוועדה , אני [מוכנה] להאריך את [ההסכם] באותם תנאים .",
         "בוקר [טוב] , אני [פותח] את הישיבה .",
         "[שלום] , אנחנו שמחים [להודיע] שחברינו [היקר] קיבל קידום ."
    ]

    words_replace_red_words = []
    sentence1_word1 = word_vectors.most_similar(positive=['לאולם', 'לדיון'], topn=3)[1]
    sentence2_word1 = word_vectors.most_similar(positive=['יכול', 'מוכנה'], negative=['גבר'], topn=3)[0]
    sentence2_word2 = word_vectors.most_similar(positive=['ההסכם'], negative=[], topn=3)[0]
    sentence3_word1 = word_vectors.most_similar(positive=['מקסים', 'טוב'], topn=3)[1]
    sentence3_word2 = word_vectors.most_similar(positive=['פותח', 'בדיון'], topn=3)[2]
    sentence4_word1 = word_vectors.most_similar(positive=['שלום', 'יקרים', 'רבותי'], topn=3)[1]
    sentence4_word2 = word_vectors.most_similar(positive=['להודיע'], topn=3)[0]
    sentence4_word3 = word_vectors.most_similar(positive=['היקר', 'המכובד', 'חברינו'], topn=3)[0]

    words_replace_red_words.append("למליאה")
    words_replace_red_words.append("יכולה")
    words_replace_red_words.append("התהליך")
    words_replace_red_words.append("בריא")
    words_replace_red_words.append("עוצר")
    words_replace_red_words.append("רבותיי")
    words_replace_red_words.append("להודיעכם")
    words_replace_red_words.append("הנאמן")

    new_sentences = []
    i = 0
    for sentence in sentences_with_red_words:
        tokens = sentence.split()
        new_sentence = []
        replaced_words = []
        for token in tokens:
            if token.startswith('[') and token.endswith(']'):
                red_word = token[1:-1]
                similar_word = words_replace_red_words[i]
                i += 1
                new_sentence.append(similar_word)
                replaced_words.append((red_word, similar_word))
            else:
                new_sentence.append(token)
        new_sentences.append((sentence, ' '.join(new_sentence), replaced_words))

    file_path = os.path.join(output_file_path, 'red_words_sentences.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, (original_sentence, new_sentence, replaced_words) in enumerate(new_sentences, 1):
            f.write(f"{i}: {original_sentence}: {new_sentence}\n")
            f.write(f"Replaced words: {', '.join([f'({orig}:{new})' for orig, new in replaced_words])}\n")
