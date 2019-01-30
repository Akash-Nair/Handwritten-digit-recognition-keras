from keras.models import load_model
import h5py
import cv2
import numpy as np

model = load_model('saved_network.h5')

#GUI and Prediction
from tkinter import *
import tkinter
import tkinter.messagebox
import urllib.request
import random
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import os

top = tkinter.Tk()
top.geometry("700x500")

name=''
#background image
backimg = ImageTk.PhotoImage(Image.open("C:/Users/akash/AppData/Local/Programs/Python/Python36/background2.jpg"))
back = Label(top,width=700,height=500,image=backimg)
back.pack()

#logo
logo = ImageTk.PhotoImage(Image.open("C:/Users/akash/AppData/Local/Programs/Python/Python36/logo.jpg"))
logo_label = Label(top,width=120,height=120,image=logo)
logo_label.place(x=10,y=10,in_=top)

#title
title = Message(top,text="Indian Institute of Information Technology, Pune",width = 700,font="Helvetica 26 bold",bg='cyan')
title.place(x=150,y=10,in_=top)

#url label
w = tkinter.Label(top,text="Image URL:",width=10,font="Helvetica 12")
w.place(x=20,y=247,in_=top)

#text field
c1=Entry(top,width=40,text=name,)
c1.place(x=130,y=250,in_=top)


def OpenFile():
    global name
    name = askopenfilename(initialdir="C:/Users/akash/AppData/Local/Programs/Python/Python36/",
                           filetypes =(("Image file", "*.jpg"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    c1.insert(10,name)

    
#browse button
file=Button(top,text="Browse",font="Helvetica 12",command=OpenFile)
file.place(x=385,y=245)

x=''
def callback():
    global url
    global x
    url = c1.get()
    x = ''
    x = ImageTk.PhotoImage(Image.open(url))
    g=Label(top,image=x,width=200,height=200,bg='white')
    g.place(x=470,y=280)

    #Prediction
    filename = ''
    for i in range(len(url)-1,-1,-1):
        if url[i] != '/':
            filename = filename + url[i]
        else:
            break

    filename = filename[::-1]
    img1 = cv2.imread(filename)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    avgr = np.average(img1,axis=0)
    avg = np.average(avgr,axis=0)
    if 255 - avg < avg:
        img1 = cv2.bitwise_not(img1)
    img1 = cv2.resize(img1,(28,28))
    img1 = np.reshape(img1,(1,28,28,1))
    img1 = img1.astype('float32')
    img1 /= 255
    
    pred = model.predict(img1)
    prediction = 'The predicted digit is: ' + str(pred.argmax())
    predict = Message(top,text=prediction,width=500,font='Helvetica 20 bold')
    predict.place(x=70,y=400,in_=top)
    
    c1.delete(0,END)
    
#upload button
b=Button(top,text="Upload",width=10,font="Helvetica 12",command=callback)
b.place(x=170,y=300,in_=top)
top.mainloop()
