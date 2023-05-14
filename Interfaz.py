import tkinter
from tkinter import filedialog
from Squat import squat_ana
from Pull_up import pull_up_ana

Patient_main = tkinter.Tk()
Patient_main.geometry("1280x720")
Patient_main.resizable("false","false")

def load_archive_squat():
    video_path = filedialog.askopenfilename(filetypes=[("all video files",".mp4")])
    squat_ana(video_path,Video_label,Info_posture,Load_button_squat)

def load_archive_pull_up():
    video_path = filedialog.askopenfilename(filetypes=[("all video files",".mp4")])
    pull_up_ana(video_path,Video_label,Info_posture,Load_button_pull_up)

#Analyzer button widgets
Info_posture = tkinter.Label(Patient_main, text="Select a video to start", font="Arial 18", fg="red")
Squat_recomendations = tkinter.Label(Patient_main, text= "Load a video preferably from a side view (sagittal plane)", font="Arial 12")
Pull_up_recomendations = tkinter.Label(Patient_main, text= "Load a video preferably from a front view (coronal plane)", font="Arial 12")
Load_button_squat = tkinter.Button(Patient_main, text="Load", command=load_archive_squat,width=10, height=1)
Load_button_pull_up = tkinter.Button(Patient_main, text="Load", command=load_archive_pull_up,width=10, height=1)
Video_label = tkinter.Label(Patient_main)


def show_squat_recomendations():
    hide_pull_up_recomendations()
    Info_posture.grid(column=5, row=0)
    Squat_recomendations.grid(column= 5, row= 1)
    Load_button_squat.grid(column=10, row=1)
    Video_label.place(x=700, y=150)

def show_pull_up_recomendations():
    hide_squat_recomendations()
    Info_posture.grid(column=5, row=0)
    Pull_up_recomendations.grid(column= 5, row= 1)
    Load_button_pull_up.grid(column=10, row=1)
    Video_label.place(x=700, y=150)

def hide_pull_up_recomendations():
    Info_posture.grid_remove()
    Load_button_pull_up.grid_remove()
    Pull_up_recomendations.grid_remove()
    Video_label.place_forget()

def hide_squat_recomendations():
    Info_posture.grid_remove()
    Load_button_squat.grid_remove()
    Squat_recomendations.grid_remove()
    Video_label.place_forget()

Buttons_background = tkinter.Canvas(Patient_main, width=280, height=720)
Buttons_background.place(x=0,y=0)

Squat_button = tkinter.Button(Patient_main, text="Squat analyzer", font="Arial 12",command=show_squat_recomendations, width=25, height=2,wraplength=200)
Pull_up_button = tkinter.Button(Patient_main, text="Pull up analyzer", font="Arial 12",command=show_pull_up_recomendations, width=25, height=2,wraplength=200)

Squat_button.grid(column=0,row=0,padx=25,pady=25)
Pull_up_button.grid(column=0,row=1)

Buttons_background.create_rectangle(0, 0, 280, 720, fill="green")

Patient_main.mainloop()
