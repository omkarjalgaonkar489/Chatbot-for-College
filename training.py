import json
import random
import numpy as np
import pickle
import tensorflow as tf
import nltk
import tflearn
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from tensorflow.python.framework import ops

#   PREPROCESSING

# Create lemmatizer object. Lemmatizer basically finds the root of any word. E.g. looking, looks, looked,
# etc. all have the root word as look
lemmatizer = WordNetLemmatizer()

# Opening json intents file
# intents = json.loads(open('Intent.json').read())
with open("Intent.json") as file:
    intents = json.load(file)

# try:
#     with open('data.pickle', 'rb') as f:
#         words, labels, training, output = pickle.load(f)
# except:
words = []  # all words in the pattern of queries go in here
labels = []  # the tags in intents file
documents_x = []  # list of lists of words in each query
documents_y = []  # corresponding tag to which words in above list belong
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents_x.append(word_list)  # [['Hi'], ['Hi', 'there']]
        documents_y.append(intent["tag"])  # ['greeting','greeting',....]

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]  # lemmatization
words = sorted(set(words))

labels = sorted(labels)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(labels, open("labels.pkl", "wb"))

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(documents_x):
    bag = []

    word_list = [lemmatizer.lemmatize(w.lower()) for w in doc]

    for w in words:
        if w in word_list:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(documents_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# with open('data.pickle', 'rb') as f:
#     pickle.dump((words, labels, training, output), f)


#   CREATE MODEL

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=300, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# def bag_of_words(s, words):
#     bag = [0 for _ in range(len(words))]
#     swds = nltk.word_tokenize(s)
#     swds = lemmatizer.lemmatize(word.lower() for word in swds)
#
#     for swd in swds:
#         for i, w in words:
#             if w == swd:
#                 bag[i] = 1
#     return np.array(bag)
#
#
# def chat():
#     global responses
#     print('Hello, How can I help? (Type quit to exit)')
#     while True:
#         inp = input('You: ')
#         if inp.lower() == 'quit':
#             break
#         results = model.predict([bag_of_words(inp, words)])
#         results_index = np.argmax(results)
#         tag = labels[results_index]
#
#         if results[results_index] > 0.6:
#             for tg in intents['intents']:
#                 if tg['tag'] == tag:
#                     responses = tg['responses']
#             print(random.choice(responses))
#         else:
#             print("I didn't understand that, try again")
#
#
# chat()
