#!/usr/bin/env python3
import random
import socket
import json
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Flatten, SpatialDropout1D
from keras.models import Sequential
import matplotlib.pyplot as plt


HOST = "127.0.0.1"
PORT = 65433


def send_stock(message, stock):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            s.send(f"{message},{stock}".encode()) #Message sent is a tuple
            data = s.recv(1024)
            print("Received:", str(data.decode()))
            s.close()
        except ConnectionRefusedError:
            print("Connection refused. Server not running?")


def send_message(message,):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            s.send(f"{message},{0}".encode()) #0 sent instead of real stock as it is not needed
            data = s.recv(1024)
            print("Received:", str(data.decode()))
            s.close()
        except ConnectionRefusedError:
            print("Connection refused. Server not running?")


def continue_loop_from_keyboard():
    answer_valid = False
    while not answer_valid:
        try:
            ans = input("Do you want to continue? (y/n): ")
        except KeyboardInterrupt:
            print("\nBye")
            return False
        if ans == "y":
            return True
        elif ans == "n":
            return False
        else:
            print("Invalid answer. Enter 'y' or 'n'.")



def main():
    #Load data from .json file
    intents = json.loads(open("intents.json").read())

    inputs = []
    classes = []
    responses = {}

    #Append data from json file to lists
    for intent in intents["intents"]:
        responses[intent["tag"]] = intent["responses"]
        for pattern in intent["patterns"]:
            inputs.append(pattern)
            classes.append(intent["tag"])

    #Convert lists to pd dataframe
    data = pd.DataFrame({"inputs":inputs, "classes":classes})

    #Remove punctuation from inputs
    data["inputs"] = data["inputs"].apply(lambda word:[letters.lower() for letters in word if letters not in string.punctuation])
    data["inputs"] = data["inputs"].apply(lambda word: ''.join(word))

    #Tokenize words and convert to np array
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data["inputs"])
    train = tokenizer.texts_to_sequences(data["inputs"])
    xtrain = pad_sequences(train) #Padding converts list of sequences to a 2D np array with length of longest sequence in list (makes all inputs uniform)

    #Encode classes to numerical values
    encoder = LabelEncoder()
    ytrain = encoder.fit_transform(data["classes"])

    #training data info
    input_size = xtrain.shape[1]
    vocab = len(tokenizer.word_index)
    output_size = encoder.classes_.shape[0]

    #creating the model
    model = Sequential()
    model.add(Embedding(input_dim = vocab+1, output_dim = 10, input_length = input_size))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(10, dropout = 0.3, recurrent_dropout = 0.3))
    model.add(Flatten())
    model.add(Dense(output_size, activation = "softmax"))

    #compile model
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    #train the model
    print("Chatbot is loading...\n")
    hist = model.fit(xtrain, ytrain, epochs=400, verbose = 0)

    continue_loop = True
    while continue_loop:
        try:
            user_input = str(input("How can I help you?\n"))
        except TypeError:
            print("Invalid input, please try again.\n")
        pred_words = []
        pred_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
        pred_input = ''.join(pred_input)
        pred_words.append(pred_input)

        pred_input = tokenizer.texts_to_sequences(pred_words)
        pred_input = np.array(pred_input).reshape(-1)
        pred_input = pad_sequences([pred_input], input_size)

        output = model.predict(pred_input, verbose = 0)
        output = output.argmax()

        tag = encoder.inverse_transform([output])[0]
        print(random.choice(responses[tag]))

        if tag == "new_stock":
            stock = input("Which stock would you like to select? ")
            send_stock(1, stock)
        elif tag == "stock_info":
            stock = input("Which stock would you like to select? ")
            send_stock(2, stock)
        elif tag == "predict_stock":
            stock = input("Which stock would you like to select? ")
            send_stock(3, stock)
        elif tag == "rundown":
            send_message(4)
        elif tag == "goodbye":
            quit()


if __name__ == "__main__":
    main()
