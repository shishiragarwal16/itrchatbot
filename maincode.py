import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import re

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('intents.json') as file:
    intents = json.load(file)

# Prepare data structures
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Data preprocessing with regex
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = re.sub(r'[^a-zA-Z\s]', '', pattern)  # Remove special characters
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [0] * len(words)
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for word in word_patterns:
        if word in words:
            bag[words.index(word)] = 1
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to array
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Build and train the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

# Load model and required data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

context = {}

def clean_up_sentence(sentence):
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)  # Remove special characters
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json, userID):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = "I'm not sure how to respond to that. Could you please provide more details?"  # Default response
    print(f"User ID: {userID}, Context: {context.get(userID)}")  # Debug statement
    for i in list_of_intents:
        if i['tag'] == tag:
            print(f"Matching intent found: {tag}")  # Debug statement
            if 'context_set' in i:
                context[userID] = i['context_set']
                print(f"Context set to: {context[userID]}")  # Debug statement
            if 'context_filter' in i:
                if userID in context and context[userID] == i['context_filter']:
                    result = random.choice(i['responses'])
                    break
            else:
                result = random.choice(i['responses'])
                break
    return result

def chatbot_response(text, userID):
    ints = predict_class(text, model)
    res = get_response(ints, intents, userID)
    return res

# Main loop to interact with the user
def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    userID = 'shishir'
    while True:
        message = input("")
        if message.lower() == 'quit':
            break
        response = chatbot_response(message, userID)
        print(response)

if __name__ == "__main__":
    chat()
