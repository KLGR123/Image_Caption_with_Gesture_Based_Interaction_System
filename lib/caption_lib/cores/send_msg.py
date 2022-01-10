import socket
import os
import sys
import struct

filepath = './saved/saved.jpg'
txtpath = './saved/history.txt'
# ip = input('input ip address:')

def send_msg():
    if os.path.exists(filepath):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('10.28.237.241', 6666)) # ip for host

            fhead = struct.pack(b'128sq', bytes(os.path.basename(filepath), encoding='utf-8'), os.stat(filepath).st_size)
            s.send(fhead)
    
            fp = open(filepath, 'rb')

            while True:
                data = fp.read(1024)
                if not data:
                    print('{} send over.'.format(filepath))
                    break
                s.send(data)
            s.close()
            os.remove('./saved/saved.jpg')

            '''
            f = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            f.connect(('10.28.238.172', 6666)) # ip for host

            fdata = open(txtpath)
            fdata = fdata.read()
            
            f.send(fdata.encode())
            f.close()
            # fdata.close()
            '''
        except socket.error as msg:
            print(msg)

        
        

