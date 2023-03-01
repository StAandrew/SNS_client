#!/usr/bin/env python3
import socket


HOST = "127.0.0.1"
PORT = 65432


def send_message(message, encoding="ascii"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            s.send(message.encode(encoding))
            data = s.recv(1024)
            print("Received:", str(data.decode(encoding)))
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
        message = "Hello, world"
        send_message(message)
        continue_loop = continue_loop_from_keyboard()

if __name__ == "__main__":
    main()
