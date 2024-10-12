import pickle

def load_model():
    with open('spam_classifier.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_message(model, vectorizer, message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)
    return prediction[0]

if __name__ == "__main__":
    model, vectorizer = load_model()

    test_messages = [
        "Привет, как дела?",
        "Зарабатывай 500$ в день, пиши в лс",
        "Давайте встретимся в 7 вечера",
        "Выиграй миллион долларов прямо сейчас!",
        "Как тебе эта шина для дрифта?"
    ]

    for message in test_messages:
        prediction = predict_message(model, vectorizer, message)
        print(f"Message: {message} | Prediction: {prediction}")
