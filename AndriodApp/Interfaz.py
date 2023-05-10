import tkinter
import cv2 as cv 
from tkinter import filedialog
import time


Patient_main = tkinter.Tk()
Patient_main.geometry("1280x720")
Patient_main.resizable("false","false")

def load_archive():
    video_path = filedialog.askopenfilename()
    video = cv.VideoCapture(video_path)

height = 120

Body_measure_condition = False

#Show body measure widgets
Info_body_measure = tkinter.Label(Patient_main, text="Actual values", font="Arial 16")
Height_tag = tkinter.Label(Patient_main, text="Height: ", font="Arial 12")
Height_value = tkinter.Label(Patient_main, text=height, font="Arial 12")

#Posture button widgets
Info_posture = tkinter.Label(Patient_main, text="Select a video to start", font="Arial 18")
Load_button = tkinter.Button(Patient_main, text="Load", command=load_archive,width=10, height=1)

def show_posture():
    hide_body_measure()
    Info_posture.place(x=730, y=25)
    Load_button.place(x=800, y=100)

def hide_posture():
    Info_posture.place_forget()
    Load_button.place_forget()


def show_body_measure():
    hide_posture()
    Info_body_measure.place(x=700, y=25)
    Height_value.place(x=350, y=200)
    Height_tag.place(x=300, y=200)

def hide_body_measure():
    Info_body_measure.place_forget()
    Height_value.place_forget()
    Height_tag.place_forget()

Buttons_background = tkinter.Canvas(Patient_main, width=280, height=720)
Buttons_background.place(x=0,y=0)

Body_measure = tkinter.Button(Patient_main, text="Modify body measure as height, weight, etc", font="Arial 12", command=show_body_measure, width=25, height=2,wraplength=200)
Posture_button = tkinter.Button(Patient_main, text="Posture analyzer", font="Arial 12",command=show_posture, width=25, height=2,wraplength=200)
Walk_button = tkinter.Button(Patient_main, text="Step analyzer", font="Arial 12", width=25, height=2,wraplength=200)

Body_measure.place(x = 20,y = 50)
Posture_button.place(x = 20, y = 120)
Walk_button.place(x = 20, y = 190)

Buttons_background.create_rectangle(0, 0, 280, 720, fill="green")

if Body_measure_condition == False:
    hide_body_measure()
    hide_posture()
    Body_measure_condition = True

Patient_main.mainloop()

"""
#Main screen definition
Main = tkinter.Tk()
Main.geometry("600x200")

patients_codes = (1512,1541521,128,25626)

#Text messages
Tag1 = tkinter.Label(Main, text = "Welcome back Dr. *********", font="Arial 22")
Tag2 = tkinter.Label(Main, text = "Please enter the patient's code ", font="Arial 16")
Tag3 = tkinter.Label(Main, text = "Patient's code does not exist, try again", font="Arial 12", bg="red")
Tag4 = tkinter.Label(Main, text = "Loading info.....", font="Arial 12")
Tag1.pack()
Tag2.pack()

#Read entry
def ReadPatient():
    patient_selection = int(Patient_selection.get())
    if not patient_selection in patients_codes:
        Tag3.pack()
        Patient_selection.delete(0,tkinter.END)
    else:
        Tag3.pack_forget()
        Patient_selection.pack_forget()
        boton1.pack_forget()
        Tag4.pack()
        Main.destroy()
        
#Definition of the button
boton1 = tkinter.Button(Main, text="Entry", command=ReadPatient)
boton1.pack(side="bottom")

#Box of text 
Patient_selection =  tkinter.Entry(Main)
Patient_selection.pack(side="bottom")
Patient_selection.bind("<Return>", lambda event: ReadPatient())

Main.mainloop()
"""