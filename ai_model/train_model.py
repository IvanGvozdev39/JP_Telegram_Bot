import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = [entry['text'] for entry in data]
    labels = [entry['label'] for entry in data]
    return texts, labels


def train_model(texts, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    return model, vectorizer


def save_model(model, vectorizer):
    import pickle
    with open('ai_model/pkl/spam_classifier.pkl', 'wb') as model_file, open('ai_model/pkl/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(model, model_file)
        pickle.dump(vectorizer, vectorizer_file)

if __name__ == "__main__":
    texts, labels = load_data('ai_model/data.json')
    model, vectorizer = train_model(texts, labels)
    save_model(model, vectorizer)
    print("Model training complete and saved.")