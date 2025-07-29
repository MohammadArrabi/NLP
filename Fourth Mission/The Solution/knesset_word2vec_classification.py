import json
import sys
import numpy as np
import random
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('must be 2 args')
        sys.exit(1)
    jsonl_file_path = sys.argv[1]
    model_file_path = sys.argv[2]

    protocols = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                protocols.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i}: {e}")

    labels = []
    for sentence in protocols:
        labels.append(sentence['protocol_type'])

    committee_sentences = []
    plenary_sentences = []
    for protocol in protocols:
        if (protocol['protocol_type'] == 'committee'):
            committee_sentences.append(protocol['sentence_text'])
        else:
            plenary_sentences.append(protocol['sentence_text'])

    committee_chunks = []
    committee_chunk_labels = []
    plenary_chunks = []
    plenary_chunk_labels = []
    chunks_size = [1,3,5]
    for chunk_size in chunks_size:
        model = Word2Vec.load(model_file_path)
        model = model.wv
        for i in range(0, len(committee_sentences) - chunk_size + 1, chunk_size):
           chunk = committee_sentences[i:i + chunk_size]
           if len(chunk) == chunk_size:
               committee_chunks.append(' '.join(chunk))
               committee_chunk_labels.append('committee')

        for i in range(0, len(plenary_sentences) - chunk_size + 1, chunk_size):
           chunk = plenary_sentences[i:i + chunk_size]
           if len(chunk) == chunk_size:
               plenary_chunks.append(' '.join(chunk))
               plenary_chunk_labels.append('plenary')

        if (len(committee_chunks) != len(plenary_chunks)):
           if (len(committee_chunks) < len(plenary_chunks)):
               plenary_chunks = random.sample(plenary_chunks, len(committee_chunks))
               plenary_chunk_labels = random.sample(plenary_chunk_labels, len(committee_chunks))
           else:
               committee_chunks = random.sample(committee_chunks, len(plenary_chunks))
               committee_chunk_labels = random.sample(committee_chunk_labels, len(plenary_chunks))

        sentences = committee_chunks + plenary_chunks
        labels = committee_chunk_labels + plenary_chunk_labels
        tokenized_sentences = []
        for sentence in sentences:
            cleaned_tokens = []
            tokens = sentence.split()
            for token in tokens:
                if 1424 <= ord(token[0]) <= 1514:
                    cleaned_tokens.append(token)
            tokenized_sentences.append(cleaned_tokens)

        sentence_embeddings = []
        for sentence in tokenized_sentences:
            sentence_vector = np.zeros(50)
            sum = 0
            for word in sentence:
                if word in model:
                    sentence_vector += model[word]
                    sum += 1
            if (sum != 0):
                sentence_vector /= sum
            sentence_embeddings.append(sentence_vector)

        features = sentence_embeddings
        KNN = KNeighborsClassifier()

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42, stratify=labels)
        KNN.fit(X_train, y_train)
        y_pred = KNN.predict(X_test)
        print('chunks size : ', chunk_size)
        print(classification_report(y_test, y_pred))