import tkinter 
import time

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
        Tag4.pack()
        Patient_selection.pack_forget()
        boton1.pack_forget()
        time.sleep(2)
        Main.destroy()
        
 
#Definition of the button
boton1 = tkinter.Button(Main, text="Entry", command=ReadPatient)
boton1.pack(side="bottom")

#Box of text 
Patient_selection =  tkinter.Entry(Main)
Patient_selection.pack(side="bottom")
Patient_selection.bind("<Return>", lambda event: ReadPatient())

Main.mainloop()
