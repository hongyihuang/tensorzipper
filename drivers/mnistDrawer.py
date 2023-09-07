import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import tkinter as tk
import scipy.misc
from skimage.draw import line_aa
from PIL import Image, ImageTk

from pathlib import Path
import requests
import pickle
import gzip
import time
import io
import os

import serial

'''
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

m = nn.AvgPool2d(2, stride=2)
x_train = m(x_train.reshape((50000, 28, 28))).reshape((50000, 14 * 14))
x_valid = m(x_valid.reshape((10000, 28, 28))).reshape((10000, 14 * 14))
'''

def exportCArray(bitstream):
  buf = io.StringIO()
  buf.write('{')
  buf.write(', '.join(map(hex, bitstream)))
  buf.write('}')
  retObj = buf.getvalue()
  buf.close()
  return retObj

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        
        self.scale = 28
        self.npframe = np.zeros((14, 14), dtype=np.uint8)
        #mnistImg = (x_train[0].reshape(14, 14).detach().numpy()*255).astype(np.uint8)
        self.upsampled = self.npframe.repeat(self.scale, axis = 0).repeat(self.scale, axis = 1)

        self.canvas = tk.Canvas(self, width=400, height=400)
        self.canvas.pack()#anchor='nw', fill='both', expand=1)

        self.npimg = ImageTk.PhotoImage(image=Image.fromarray(self.upsampled))
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.npimg)

        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # create elements
        self.label = tk.Label(self, text="Waiting...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Run", command = self.classify)
        self.clear_btn = tk.Button(self, text="Clear", command = self.clear)
        # create grid
        self.canvas.grid(row=0, column=0, pady=2)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=0, pady=2)

        self.ser = serial.Serial('/dev/tty.usbserial-112201', 9600, timeout = None)  # open serial port
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        print("Currently Using: ", self.ser.name)         # check which port was really used

    def __del__(self):
        self.ser.close()             # close port

    def get_x_and_y(self, event):
        global lasx, lasy
        lasx, lasy = event.x, event.y

    def update_img(self):
        upsampled = self.npframe.repeat(self.scale, axis = 0).repeat(self.scale, axis = 1)

        self.npimg = ImageTk.PhotoImage(image=Image.fromarray(upsampled))
        self.canvas.itemconfig(self.canvas_img, image=self.npimg)

    def draw(self, event):
        global lasx, lasy
        #canvas.create_line((lasx, lasy, event.x, event.y), 
        #                   fill='white', width=5)
        rr, cc, val = line_aa(  lasy//self.scale, lasx//self.scale,
                                event.y//self.scale, event.x//self.scale)
        self.npframe[rr, cc] = (val * 255).astype(np.uint8)
        #print(rr, cc, (val * 255).astype(np.uint8))
        #print(npframe)
        self.update_img()

        #imageref to prevent deletion
        #self.canvas.imgref = img
        lasx, lasy = event.x, event.y

    def classify(self):
        arr = exportCArray(self.npframe.flatten() >> 6)
        print(arr)
        print(len(arr))

        cmd = "riscv64-unknown-elf-gdb --command=./script.gdb /Users/hongyihuang/Documents/GitHub/BearlyML/Baremetal-IDE/workspace/build/firmware.elf > out.txt"
        f = open("script.gdb", "w") #overwrite
        f.write("target extended-remote localhost:3333\n")
        #f.write("load /Users/hongyihuang/Documents/GitHub/BearlyML/Baremetal-IDE/workspace/build/firmware.elf\n")
        f.write("set $pc=0x20000000\n")
        f.write("b getMNIST_UART\n")
        f.write("run\n")
        f.write("step\n")
        f.write("step\n")
        f.write("set var buf = " + arr + "\n")
        f.write("continue\n")
        f.write("quit\n")
        f.close()

        # run script
        os.system(cmd)

        rdy = self.ser.read(3)
        print(rdy)        

        # recieve inference result
        result = self.ser.read(1)
        while (len(result) != 1):
            result = self.ser.read(1)
        print("Result: ")
        print(result)
        self.label.configure(text = str(result[0]))

    def clear(self):
        self.npframe = np.zeros((14, 14), dtype=np.uint8)
        self.label.configure(text="Waiting")
        self.update_img()

app = App()#tk.Tk()

#app.geometry("400x400")

app.mainloop()
