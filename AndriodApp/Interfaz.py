import tkinter as tk

class Main:
    def __init__(self, master):
        self.master = master
        self.master.configure(bg = "blue")
        master.title("MABOAMDS")
        master.geometry("1280x720")
        master.resizable(False, False)

        #boton
        self.btn1 = tk.Button(master, text="Análisis de postura", command=self.posture)
        self.btn1.pack(side = "right", padx = 30)

        #boton
        self.btn2 = tk.Button(master, text="Análisis de carrera", command=self.running)
        self.btn2.pack(side = "left", padx = 30)

    def posture(self):
        self.master.destroy()

        posture_window = tk.Toplevel()
        mi_ventana1 = Posture_window(posture_window)

    def running(self):
        self.master.destroy()

        runnig_window = tk.Toplevel()
        mi_ventana2 = Runnig_window(runnig_window)


class Running_window:
    def __init__(self, master):
        self.master = master
        master.title("Running analysis")

        label1 = tk.Label(master, text="Proximamente")
        label1.pack()

class Posture_window:
    def __init__(self, master):
        self.master = master
        master.title("Posture analysis")

        label2 = tk.Label(master, text="solo en cines")
        label2.pack()

root = tk.Tk()
mi_ventana_principal = Main(root)

root.mainloop()
