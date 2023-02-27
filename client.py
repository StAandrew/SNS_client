#!/usr/bin/env python3

import socket

def Main():


    HOST = '127.0.0.1'
    PORT = 65432

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))

            message = "Hello, world"
            s.send(message.encode("ascii"))
            
            data = s.recv(1024)

        print('Received:', str(data.decode("ascii")))

        ans = input("\nDo you want to continue? (y/n): ")
        if ans == 'y':
            continue
        else:
            break

    s.close()

if __name__ == '__main__':
    Main()
