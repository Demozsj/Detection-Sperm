import tkinter as tk
from tkinter.filedialog import *
from get_predicted_result import TODCNN
from PIL import Image
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter.messagebox import showinfo

global img_path, vid_path, tod_cnn, capture


def load_model():
    iou = text1.get(1.0, tk.END)
    con = text2.get(1.0, tk.END)
    model_path = text3.get(1.0, tk.END)
    model_path = model_path[:-1]
    if len(iou) > 1:
        iou = float(iou)
    else:
        iou = 0.5
    if len(con) > 1:
        confidence = float(con)
    else:
        confidence = 0.5
    if len(model_path) < 1:
        model_path = 'model_data/sperm_last.h5'
    global tod_cnn
    tod_cnn = TODCNN(model_path, iou, confidence)


def Choose_Image():
    global img_path
    img_path = askopenfilename()
    img = Image.open(img_path)
    imgtk = ImageTk.PhotoImage(image=img)
    L0.place(x=10, y=10)
    L0.imgtk = imgtk
    L0.config(image=imgtk)
    L1.place_forget()


def detect_img():
    L0.place_forget()
    try:
        image = Image.open(img_path)
    except:
        print('Open Error!')
    else:
        result = tod_cnn.detect_image_for_GUI(image)
    save_path = img_path[:-4] + '_detection' + img_path[-4:]
    result.save(save_path)
    imgtk = ImageTk.PhotoImage(image=result)
    L1.place(x=10, y=10)
    L1.imgtk = imgtk
    L1.config(image=imgtk)


def save_video(img_list):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_path_det = vid_path[:-4] + '_detection' + vid_path[-4:]
    videoWriter = cv2.VideoWriter(save_path_det, fourcc, fps, img_list[0].size)
    for i in range(len(img_list)):
        img_frame = cv2.cvtColor(np.array(img_list[i]), cv2.COLOR_RGB2BGR)
        videoWriter.write(img_frame)
    videoWriter.release()
    cv2.destroyAllWindows()


def Choose_Video():
    L1.place_forget()
    global vid_path
    vid_path = askopenfilename()


def detect_video():
    global capture
    capture = cv2.VideoCapture(vid_path)
    l0 = tk.Label(window)
    l0.place(x=10, y=10)
    img_list=[]
    video_loop(l0, img_list)


def video_loop(panela, img_list):
    success, img = capture.read()
    if success:
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        current_image = tod_cnn.detect_image_for_GUI(current_image)
        img_list.append(current_image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panela.imgtk = imgtk
        panela.config(image=imgtk)
        window.after(1, lambda: video_loop(panela, img_list))
    else:
        panela.place_forget()
        save_video(img_list)


window = tk.Tk()
window.title('TOD-CNN (A Convolutional Neural Network for Tiny Object Detection)')
window.geometry("718x650")

tk.Label(window, text="IoU Threshold:").place(x=30, y=549)
tk.Label(window, text="Confidence:").place(x=170, y=549)
tk.Label(window, text="Model Path:").place(x=295, y=549)

text1 = tk.Text(window, width=5, height=1)
text1.place(x=125, y=550)
text2 = tk.Text(window, width=5, height=1)
text2.place(x=248, y=550)
text3 = tk.Text(window, width=25, height=1)
text3.place(x=375, y=550)

button0 = tk.Button(window, text='Choose Image', width=20, height=1, command=Choose_Image)
button0.place(x=30, y=600)
button1 = tk.Button(window, text='Detection Image', width=20, height=1, command=detect_img)
button1.place(x=200, y=600)
button2 = tk.Button(window, text='Choose Video', width=20, height=1, command=Choose_Video)
button2.place(x=370, y=600)
button3 = tk.Button(window, text='Detection Video', width=20, height=1, command=detect_video)
button3.place(x=540, y=600)
button3 = tk.Button(window, text='Load Model', width=15, height=1, command=load_model)
button3.place(x=575, y=545)

L0 = tk.Label(window)
L1 = Label(window)

window.mainloop()
