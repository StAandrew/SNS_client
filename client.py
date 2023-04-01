#!/usr/bin/env python3
import random
import socket
import json
import string
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Flatten, SpatialDropout1D
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
from config import figures_dir, dataset_dir

HOST = "127.0.0.1"
PORT = 65432

def get_dir(ticker):
    stock_csv = ticker + "_data.csv"
    stock_dir = os.path.join(dataset_dir, stock_csv)
    return stock_dir


def send_stock(stock):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            stock_pickle = pickle.dumps(stock)
            s.send(stock_pickle) #Message sent as a pickle object
            data_pickle = s.recv(1024)
            data = pickle.loads(data_pickle)
            #print("Received:", str(data.decode()))
            s.close()
            return data
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

def plot_prediction(ticker, predictions):
    stock_dir = get_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)
    plt.plot(dataset.tail(60).index, dataset.tail(60)['Close'], color = 'red', label = f'Historical {ticker} Price')
    plt.plot(predictions.index, predictions['Close'], color = 'blue', label = f'Predicted {ticker} Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, ticker + "_prediction.png"))
    plt.show()

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

        stockList = []
        if tag == "get_price":
            stock = input("Which stock would you like to select? ")
            days = input("How many days ahead should I predict? ")
            stockList.append(stock)
            stockList.append(days)
            stockList.append('1')
            data_rec = send_stock(stockList)
            print(f"The closing price of {stock} is ${data_rec}\n")

        elif tag == "get_daily_return":
            valid = False
            num = 1
            while valid == False:
                stock = input(f"Please enter stock {num} (or type 'Done' to finish): ")
                if stock.lower() == "done":
                    valid = True
                else:
                    stockList.append(stock)
                    num += 1
            days = input("How many days ahead should I predict? ")
            stockList.append(days)
            data_rec = []
            if len(stockList) == 2: #Checks that the list only contains one stock (and the number of days to predict)
                stockList.append('2')
                data_rec = send_stock(stockList)
            else:
                stockList.append('6')
                data_rec = send_stock(stockList)
            
            print(f"The daily returns for your chosen stocks are:\n{data_rec} ")

        elif tag == "get_avg_return":
            stock = input("Which stock would you like to select? ")
            days = input("How many days ahead should I predict? ")
            stockList.append(stock)
            stockList.append(days)
            stockList.append('3')
            data_rec = send_stock(stockList)
            print(f"The average return of {stock} over {days} days is ${data_rec}\n")

        elif tag == "get_std":
            stock = input("Which stock would you like to select? ")
            days = input("How many days ahead should I predict? ")
            stockList.append(stock)
            stockList.append(days)
            stockList.append('4')
            data_rec = send_stock(stockList)
            print(f"The volatility of {stock} over {days} days is {data_rec}\n")

        elif tag == "get_sharpe":
            stock = input("Which stock would you like to select? ")
            days = input("How many days ahead should I predict? ")
            rfr = input("What risk free rate would you like to use? (between 0 and 1): ")
            stockList.append(stock)
            stockList.append(rfr) #appended before days and choice so that the server can process the data correctly
            stockList.append(days)
            stockList.append('5')
            data_rec = send_stock(stockList)
            print(f"The sharpe ratio of {stock} over {days} days is {data_rec}\n")

        elif tag == "optimise":
            valid = False
            num = 1
            while valid == False:
                stock = input(f"Please enter stock {num} (or type 'Done' to finish): ")
                if stock.lower() == "done":
                    valid = True
                else:
                    stockList.append(stock)
                    num += 1
            days = input("How many days ahead should I predict? ")
            
            valid = False
            while valid == False:
                try:
                    optimise_type = int(input("Would you like to optimise for 1. minimum variance or 2. maximum sharpe ratio? (please type 1 or 2 to select option): "))
                except TypeError:
                    print("Invalid input, please try again.\n")
                if optimise_type == 1 or optimise_type == 2:
                    valid = True
                else:
                    print(f"{optimise_type} is not a valid input, please enter 1 or 2 accordingly.\n")
            if optimise_type == 1:
                stockList.append(days)
                stockList.append('7')
                data_rec = send_stock(stockList)
                min_var = data_rec.pop()

                print(f"Here are the optimised weights: {data_rec}")
                print(f"This produces a variance of {min_var}")
            else:
                rfr = input("What risk free rate would you like to use? (between 0 and 1): ")
                stockList.append(rfr)
                stockList.append(days)
                stockList.append('8')
                data_rec = send_stock(stockList)
                max_sharpe = data_rec.pop()

                print(f"Here are the optimised weights: {data_rec}")
                print(f"This produces a sharpe ratio of {max_sharpe}")

            



        elif tag == "goodbye":
            quit()


if __name__ == "__main__":
    main()
