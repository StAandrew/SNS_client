#!/usr/bin/env python3
import socket


HOST = "127.0.0.1"
PORT = 65432


def main():
    continue_loop = True
    while continue_loop:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))

            message = "Hello, world"
            s.send(message.encode("ascii"))

            data = s.recv(1024)
        print("Received:", str(data.decode("ascii")))
        answer_valid = False
        while not answer_valid:
            ans = input("Do you want to continue? (y/n): ")
            if ans == "y":
                continue_loop = True
                answer_valid = True
            elif ans == "n":
                continue_loop = False
            else:
                print("Invalid answer. Enter 'y' or 'n'.")
    s.close()


if __name__ == "__main__":
    main()
