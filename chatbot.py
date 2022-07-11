import random
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
import tflearn
from training import model
import tensorflow
import datetime

lemmatizer = WordNetLemmatizer()
with open("Intent.json") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
# model = model.load('model.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    swds = nltk.word_tokenize(s)
    swds = [lemmatizer.lemmatize(w.lower()) for w in swds]

    for swd in swds:
        for i, w in enumerate(words):
            if w == swd:
                bag[i] = 1
    return np.array(bag)


def chat():
    global responses
    print("Hi. What can I do for you? (Type quit to exit)")
    while True:
        inp = input("YOU: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.6:
            for tg in intents["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    print(random.choice(responses))

                    if tag == "TimeQuery":
                        time = f"{datetime.datetime.now().time().hour}:{datetime.datetime.now().time().minute}"
                        print(random.choice(tg["extension"]["responses"]) % time)

                    if tag == "CourseQuery":
                        if input("YOU: ") == "1":
                            print(tg["extension"]["responses"]["1"])
                        elif input("YOU: ") == "2":
                            print(tg["extension"]["responses"]["2"])
                        elif input("YOU: ") == "3":
                            print(tg["extension"]["responses"]["3"])
                        elif input("YOU: ") == "4":
                            print(tg["extension"]["responses"]["4"])
                        print("To find out more, visit https://vit.edu.in/admissions.html")

                    if tag == "GoodBye" or tag == "CourtesyGoodBye":
                        return

                    break

        else:
            print("I didn't get that. Please try again.")


chat()
