#!/usr/bin/env python3
import random
import socket
import json
import string
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Flatten, SpatialDropout1D
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
from config import figures_dir, dataset_dir

HOST = "127.0.0.1"
PORT = 65433

def isTicker(t: str) -> bool: #Finction to check if ticker/stock name exists on yahoo finance
    ticker = yf.Ticker(t)
    info = ticker.history(period='7d', interval='1d')
    return len(info) > 0

def isfloat(num): #Function to return true if string is a float
    try:
        float(num)
        return True
    except ValueError:
        return False

def get_dir(ticker):
    stock_csv = ticker + "_data.csv"
    stock_dir = os.path.join(dataset_dir, stock_csv)
    return stock_dir


def send_stock(stock):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            stock_pickle = pickle.dumps(stock) #Convert list into pickle object (easy to send via socket)
            s.send(stock_pickle)
            data_pickle = s.recv(1024) #Recieve data from server
            data = pickle.loads(data_pickle) #'unpickle' data into a list
            s.close() #close connection
            return data
        except ConnectionRefusedError:
            print("Connection refused. Server not running?")



def main():

    ###### MACHINE LEARNING ALGORITHM ######

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
    print("\nChatbot is loading...\n")
    hist = model.fit(xtrain, ytrain, epochs=400, verbose = 0)



    ###### USER INTERFACE/SERVER COMMUNICATION ######

    continue_loop = True
    while continue_loop:
        try:
            user_input = str(input("How can I help you?\n"))
        except TypeError:
            print("Invalid input, please try again.\n")

        ###Convert user input to same format as training data ###
        #Convert to lower case
        pred_words = []
        pred_input = [letters.lower() for letters in user_input if letters not in string.punctuation] 
        pred_input = ''.join(pred_input)
        pred_words.append(pred_input)

        #Tokenize the words
        pred_input = tokenizer.texts_to_sequences(pred_words)
        pred_input = np.array(pred_input).reshape(-1)
        pred_input = pad_sequences([pred_input], input_size)

        
        #use the model to find most likely tag
        output = model.predict(pred_input, verbose = 0)
        output = output.argmax()
        tag = encoder.inverse_transform([output])[0]

        #Print a random response stored in the .json file (only used for greetings and goodbyes)
        print(random.choice(responses[tag]))


        ### ACTIONS FOR EACH TAG ###
        """""""""""""""
        The client will send the relevant data, followed by a number (1-8) to the server in the form of a list. The number at the end of the list
        determins the function that the server will use.
        """""""""""""""

        stockList = []

        #Find the price of a stock at a given day
        if tag == "get_price":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? ")
                if(isTicker(stock) and days.isdigit()):
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")

            stockList.append(stock)
            stockList.append(days)
            stockList.append('1')
            data_rec = round(send_stock(stockList), 2)
            print(f"The closing price of {stock} {days} days from now will be ${data_rec}\n")

        
        #Find the daily returns of a stock/portfolio until a given day
        elif tag == "get_daily_return":
            valid = False
            num = 1
            while valid == False:
                stock = input(f"Please enter stock {num} (or type 'Done' to finish): ")
                if stock.lower() == "done":
                    valid = True
                elif not isTicker(stock):
                    print("Invalid input. Please try again.\n")
                else:
                    stockList.append(stock)
                    num += 1
            valid = False
            while valid == False:
                days = input("How many days ahead should I predict? ")
                if days.isdigit():
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            stockList.append(days)
            data_rec = []
            #Check if user wants to predict a single stock or multiple, and select the relevent choice
            if len(stockList) == 2: #Checks that the list only contains one stock (and the number of days to predict)
                stockList.append('2')
                data_rec = send_stock(stockList)
                data_round = [round(price, 2) for price in data_rec]
                print(f"The daily returns of {stockList[0]} over {days} days will be: {data_round}\n")
            else: #List contains more than one stock and no. of days, so must contain multiple stocks
                stockList.append('6') 
                data_rec = np.round(send_stock(stockList), 2)
                print(f"The daily returns for your chosen stocks over {days} days will be:")
                for i in range (len(stockList)-2):
                    print(f"{stockList[i]}: {np.transpose(data_rec[:,i])}\n")
            


        #Find the average return of a stock over a number of days
        elif tag == "get_avg_return":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? ")
                if(isTicker(stock) and days.isdigit()):
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            stockList.append(stock)
            stockList.append(days)
            stockList.append('3')
            data_rec = round(send_stock(stockList), 2)
            print(f"The average return of {stock} over {days} days will be ${data_rec}\n")
            

            
        #Find the volatility of a stock over a number of days
        elif tag == "get_std":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? ")
                if(isTicker(stock) and days.isdigit()):
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            stockList.append(stock)
            stockList.append(days)
            stockList.append('4')
            data_rec = round(send_stock(stockList), 2)
            print(f"The volatility of {stock} over {days} days will be {data_rec}\n")


        #Find the sharpe ratio of a stock at a given rfr over a number of days
        elif tag == "get_sharpe":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? ")
                rfr = input("What risk free rate would you like to use? ")
                if(isTicker(stock) and days.isdigit() and isfloat(rfr)):
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            
            stockList.append(stock)
            stockList.append(rfr) #appended before days and choice so that the server can process the data correctly
            stockList.append(days)
            stockList.append('5')
            data_rec = round(send_stock(stockList), 2)
            print(f"The sharpe ratio of {stock} over {days} days will be {data_rec}\n")


        #Optimise a portfolio of stocks (find the optimal ratio to split the investment) to minimise variance or maximise sharp ratio (chosen by user)
        elif tag == "optimise":
            #Stock selection
            valid = False
            num = 1
            while valid == False:
                stock = input(f"Please enter stock {num} (or type 'Done' to finish): ")
                if stock.lower() == "done":
                    valid = True
                elif not isTicker(stock):
                    print("Invalid input. Please try again.\n")
                else:
                    stockList.append(stock)
                    num += 1
            
            valid = False
            while valid == False:
                days = input("How many days ahead should I predict? ")
                if days.isdigit():
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            
            #Optimisation type selection
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

            #Minimum variance optimisation actions
            if optimise_type == 1:
                stockList.append(days)
                stockList.append('7')
                data_rec = send_stock(stockList)
                min_var = data_rec.pop()

                print(f"The optimal ratio to split your investment is as follows:")

                for i in range (len(stockList)-2):
                    print(f"{stockList[i]}: {round(data_rec[i]*100, 2)}%")
                print(f"This produces a variance of {min_var}")

            #Maximum sharpe ratio actions
            else:
                valid = False
                while valid == False:
                    rfr = input("What risk free rate would you like to use?: ")
                    if isfloat(rfr):
                        valid = True
                    else:
                        print("Invalid input. Please try again.\n")
                
                stockList.append(rfr)
                stockList.append(days)
                stockList.append('8')
                data_rec = send_stock(stockList)
                max_sharpe = data_rec.pop()

                print(f"The optimal ratio to split your investment is as follows:")

                for i in range (len(stockList)-3):
                    print(f"{stockList[i]}: {round(data_rec[i]*100, 2)}%")
                print(f"This produces a sharpe ratio of {max_sharpe}")

        #Stop the program is the user says goodbye
        elif tag == "goodbye":
            quit()


if __name__ == "__main__":
    main()
