import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import io

# load the pre-trained model
model = tf.keras.models.load_model('handwritten.h5')

# define the canvas size
canvas_width = 280
canvas_height = 280

def paint(event):
    # draw on the canvas
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

def clear():
    # clear the canvas
    canvas.delete("all")
    # create a new image
    global image
    image = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

def recognize_digit():
    # get the canvas image
    canvas_image = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(canvas_image.encode('utf-8')))
    # recognize the digit using the pre-trained model
    digit, prob = predict_digit(img, model)
    # display the result
    result_label.config(text=f"Predicted digit: {digit}\nProbability: {prob:.2f}")

def predict_digit(img, model):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    img = np.abs(img-1)
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

# create the main window and canvas
root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack(expand='yes', fill='both')

# create a new image to draw on
image = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
draw = ImageDraw.Draw(image)

# bind mouse events to the canvas
canvas.bind("<B1-Motion>", paint)

# create buttons for clearing and recognizing the digit
clear_button = tk.Button(root, text="Clear", command=clear)
clear_button.pack(side='left', padx=10)

recognize_button = tk.Button(root, text="Recognize", command=recognize_digit)
recognize_button.pack(side='right', padx=10)

# create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# start the main event loop
root.mainloop()
