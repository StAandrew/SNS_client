#!/usr/bin/env python3
import socket


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
    continue_loop = True
    while continue_loop:
        try:
            choice = int(input("Please select one of the following options:\n 1. Fetch Stock Data\n 2. Predict stock price\n 3. Daily summary\n"))
        except TypeError:
            print("Invalid input, please try again.\n")
        if choice == 1 or choice == 2:
            stock = input("Which stock would you like to select? ")
            send_stock(choice, stock)
        elif choice == 3:
            send_message(choice)
        else:
            print("Invalid input, please try again.\n")

        continue_loop = continue_loop_from_keyboard()

if __name__ == "__main__":
    main()
