import json
import sys
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split , cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def create_chunks(sentences, chunk_size):
    chunks = []
    for i in range(0, len(sentences) - chunk_size + 1, chunk_size):
        chunk = sentences[i:i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(' '.join(chunk))
    return chunks

def evaluate_chunk_size(chunk_size):
    committee_chunks = create_chunks(committee_sentences, chunk_size)
    plenary_chunks = create_chunks(plenary_sentences, chunk_size)

    min_size = min(len(committee_chunks), len(plenary_chunks))
    committee_chunks = random.sample(committee_chunks, min_size)
    plenary_chunks = random.sample(plenary_chunks, min_size)

    chunks = committee_chunks + plenary_chunks
    labels = ['committee'] * len(committee_chunks) + ['plenary'] * len(plenary_chunks)

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(chunks)

    X_custom = choose_MyFeatures(chunks)

    knn = KNeighborsClassifier()
    lr = LogisticRegression(max_iter=1000)

    knn_accuracy_tfidf, knn_report_tfidf = k_fold_cross_validation(knn, X_tfidf, labels)
    lr_accuracy_tfidf, lr_report_tfidf = k_fold_cross_validation(lr, X_tfidf, labels)

    knn_accuracy_custom, knn_report_custom = k_fold_cross_validation(knn, X_custom, labels)
    lr_accuracy_custom, lr_report_custom = k_fold_cross_validation(lr, X_custom, labels)

    return {
        'chunk_size': chunk_size,
        'knn_accuracy_tfidf': knn_accuracy_tfidf,
        'lr_accuracy_tfidf': lr_accuracy_tfidf,
        'knn_accuracy_custom': knn_accuracy_custom,
        'lr_accuracy_custom': lr_accuracy_custom,
        'knn_report_tfidf': knn_report_tfidf,
        'lr_report_tfidf': lr_report_tfidf,
        'knn_report_custom': knn_report_custom,
        'lr_report_custom': lr_report_custom
    }

def analyze_results(results):
    best_result = max(results,
                      key=lambda x: max(x['knn_accuracy_tfidf'], x['lr_accuracy_tfidf'], x['knn_accuracy_custom'],
                                        x['lr_accuracy_custom']))
    ideal_chunk_size = best_result['chunk_size']
    '''
    print(f"The ideal chunk size is {ideal_chunk_size} with the following accuracies:")
    print(f"KNN Accuracy (TF-IDF): {best_result['knn_accuracy_tfidf']}")
    print(f"Logistic Regression Accuracy (TF-IDF): {best_result['lr_accuracy_tfidf']}")
    print(f"KNN Accuracy (Custom Features): {best_result['knn_accuracy_custom']}")
    print(f"Logistic Regression Accuracy (Custom Features): {best_result['lr_accuracy_custom']}")
    '''
    return ideal_chunk_size
def k_fold_cross_validation(model, X, y, cv=5):
    y_pred = cross_val_predict(model, X, y, cv=cv)
    report = classification_report(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return accuracy , report
def fun_eval_train_test_split(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, report
def choose_MyFeatures(chunks):
    MyFeatures_list = []
    content_words = {'ועדה', 'ועדת', 'חברי הכנסת', 'חבר הכנסת' , 'אדוני היושב-ראש' ,'מי נגד' , 'הצבעה' , 'מי בעד','הצעת','הצעה','אדוני','חבר','חברי'}
    for chunk in chunks:
        sentences = chunk.split('.')
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
        features = [len(chunk.split()), avg_sentence_length]
        for word in content_words:
            features.append(chunk.count(word))
        MyFeatures_list.append(features)

    return np.array(MyFeatures_list)

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('must be 3 args')
        sys.exit(1)
    jsonl_file_path = sys.argv[1]
    knesset_text_chunks_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    random.seed(42)
    np.random.seed(42)

    # שלב 1
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

    # שלב 2
    committee_sentences = []
    plenary_sentences = []
    for protocol in protocols:
        if(protocol['protocol_type'] == 'committee'):
            committee_sentences.append(protocol['sentence_text'])
        else:
            plenary_sentences.append(protocol['sentence_text'])

    committee_chunks = []
    committee_chunk_labels = []
    plenary_chunks = []
    plenary_chunk_labels = []
    chunk_size = 5
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


    # שלב 3
    '''
    print(f"Number of committee chunks before balancing: {len(committee_chunks)}")
    print(f"Number of plenary chunks before balancing: {len(plenary_chunks)}")
    '''

    if(len(committee_chunks) != len(plenary_chunks)):
        if(len(committee_chunks) < len(plenary_chunks)):
            plenary_chunks = random.sample(plenary_chunks, len(committee_chunks))
            plenary_chunk_labels = random.sample(plenary_chunk_labels, len(committee_chunks))
        else:
            committee_chunks = random.sample(committee_chunks, len(plenary_chunks))
            committee_chunk_labels = random.sample(committee_chunk_labels, len(plenary_chunks))

    '''
    print(f"Number of committee sentences after balancing: {len(committee_chunks)}")
    print(f"Number of plenary sentences after balancing: {len(plenary_chunks)}")
    '''

    chunks = committee_chunks + plenary_chunks
    labels = committee_chunk_labels + plenary_chunk_labels

    # שלב 4
    tfidf_vectorizer_model = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer_model.fit_transform(chunks)
    X_MyFeatures = choose_MyFeatures(chunks)

    #שלב 5
    KNN = KNeighborsClassifier()
    Logistic_Regression = LogisticRegression(max_iter=1000)


    knn_fcv_tfidf_accuracy , knn_fcv_tfidf_report = k_fold_cross_validation(KNN, X_tfidf, labels)
    logistic_regression_fcv_tfidf_accuracy , logistic_regression_fcv_tfidf_report = k_fold_cross_validation(Logistic_Regression, X_tfidf, labels)

    knn_fcv_MyFeatures_accuracy , knn_fcv_MyFeatures_report = k_fold_cross_validation(KNN, X_MyFeatures, labels)
    logistic_regression_fcv_MyFeatures_accuracy , logistic_regression_fcv_MyFeatures_report = k_fold_cross_validation(Logistic_Regression, X_MyFeatures, labels)

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, labels, test_size=0.1, random_state=42, stratify=labels)
    X_train_MyFeatures, X_test_MyFeatures, y_train_MyFeatures, y_test_MyFeatures = train_test_split(X_MyFeatures, labels, test_size=0.1, random_state=42, stratify=labels)

    knn_tts_tfidf_accuracy , knn_tts_tfidf_report = fun_eval_train_test_split(KNN, X_train_tfidf, y_train_tfidf, X_test_tfidf , y_test_tfidf)
    logistic_regression_tts_tfidf_accuracy , logistic_regression_tts_tfidf_report = fun_eval_train_test_split(Logistic_Regression, X_train_tfidf, y_train_tfidf, X_test_tfidf , y_test_tfidf)

    knn_tts_MyFeatures_accuracy , knn_tts_MyFeatures_report = fun_eval_train_test_split(KNN, X_train_MyFeatures, y_train_MyFeatures, X_test_MyFeatures , y_test_MyFeatures)
    logistic_regression_tts_MyFeatures_accuracy , logistic_regression_tts_MyFeatures_report = fun_eval_train_test_split(Logistic_Regression, X_train_MyFeatures, y_train_MyFeatures, X_test_MyFeatures , y_test_MyFeatures)


    # Print Classification Reports
    '''
    print("KNN tfidf 5-fold_cross_validation Classification Report:")
    print(knn_fcv_tfidf_report)

    print("Logistic Regression tfidf 5-fold_cross_validation Classification Report:")
    print(logistic_regression_fcv_tfidf_report)

    print("KNN My_Features 5-fold_cross_validation Classification Report:")
    print(knn_fcv_MyFeatures_report)

    print("Logistic Regression My_Features 5-fold_cross_validation Classification Report:")
    print(logistic_regression_fcv_MyFeatures_report)

    print("KNN tfidf Train_Test_Split Classification Report:")
    print(knn_tts_tfidf_report)

    print("Logistic Regression tfidf Train_Test_Split Classification Report:")
    print(logistic_regression_tts_tfidf_report)

    print("KNN My_Features Train_Test_Split Classification Report:")
    print(knn_tts_MyFeatures_report)

    print("Logistic Regression My_Features Train_Test_Split Classification Report:")
    print(logistic_regression_tts_MyFeatures_report)
    '''

    # שלב 6
    best_model = None
    best_accuracy = 0
    best_vectorizer = None


    if logistic_regression_fcv_tfidf_accuracy > knn_fcv_tfidf_accuracy:
        best_fcv_model_tfidf = Logistic_Regression
        best_fcv_accuracy_tfidf = logistic_regression_fcv_tfidf_accuracy
    else:
        best_fcv_model_tfidf = KNN
        best_fcv_accuracy_tfidf = knn_fcv_tfidf_accuracy

    if logistic_regression_tts_tfidf_accuracy > knn_tts_tfidf_accuracy:
        best_tts_model_tfidf = Logistic_Regression
        best_tts_accuracy_tfidf = logistic_regression_tts_tfidf_accuracy
    else:
        best_tts_model_tfidf = KNN
        best_tts_accuracy_tfidf = knn_tts_tfidf_accuracy


    if logistic_regression_fcv_MyFeatures_accuracy > knn_fcv_MyFeatures_accuracy:
        best_fcv_model_MyFeatures = Logistic_Regression
        best_fcv_accuracy_MyFeatures = logistic_regression_fcv_MyFeatures_accuracy
    else:
        best_fcv_model_MyFeatures = KNN
        best_fcv_accuracy_MyFeatures = knn_fcv_MyFeatures_accuracy

    if logistic_regression_tts_MyFeatures_accuracy > knn_tts_MyFeatures_accuracy:
        best_tts_model_MyFeatures = Logistic_Regression
        best_tts_accuracy_MyFeatures = logistic_regression_tts_MyFeatures_accuracy
    else:
        best_tts_model_MyFeatures = KNN
        best_tts_accuracy_MyFeatures = knn_tts_MyFeatures_accuracy

    # בחירת המודל הסופי
    if best_fcv_accuracy_tfidf > best_tts_accuracy_tfidf:
        best_model_tfidf = best_fcv_model_tfidf
        best_accuracy_tfidf = best_fcv_accuracy_tfidf
    else:
        best_model_tfidf = best_tts_model_tfidf
        best_accuracy_tfidf = best_tts_accuracy_tfidf

    if best_fcv_accuracy_MyFeatures > best_tts_accuracy_MyFeatures:
        best_model_MyFeatures = best_fcv_model_MyFeatures
        best_accuracy_MyFeatures = best_fcv_accuracy_MyFeatures
    else:
        best_model_MyFeatures = best_tts_model_MyFeatures
        best_accuracy_MyFeatures = best_tts_accuracy_MyFeatures

    if best_accuracy_tfidf > best_accuracy_MyFeatures:
        best_model = best_model_tfidf
        best_vectorizer = tfidf_vectorizer_model
        best_features = X_tfidf
    else:
        best_model = best_model_MyFeatures
        best_features = X_MyFeatures

    best_model.fit(best_features, labels)

    chunks_to_classify = []
    with open(knesset_text_chunks_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks_to_classify.append(line.strip())

    if best_vectorizer is not None:
        X_chunks_to_classify = best_vectorizer.transform(chunks_to_classify)
    else:
        X_chunks_to_classify = choose_MyFeatures(chunks_to_classify)

    predictions = best_model.predict(X_chunks_to_classify)

    output_file_path = os.path.join(output_file_path, 'classification_results.txt')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(prediction + "\n")

    results = []
    chunk_sizes = [1, 3, 5, 10, 20]
    for size in chunk_sizes:
        result = evaluate_chunk_size(size)
        results.append(result)
    '''
    for result in results:
        print(f"Chunk Size: {result['chunk_size']}")
        print(f"KNN Accuracy (TF-IDF): {result['knn_accuracy_tfidf']}")
        print(f"Logistic Regression Accuracy (TF-IDF): {result['lr_accuracy_tfidf']}")
        print(f"KNN Accuracy (Custom Features): {result['knn_accuracy_custom']}")
        print(f"Logistic Regression Accuracy (Custom Features): {result['lr_accuracy_custom']}")
        print("")
    '''
    ideal_chunk_size = analyze_results(results)