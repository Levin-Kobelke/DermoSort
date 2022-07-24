import tkinter as tk
from tkinter import Button, Label, filedialog
import os
import sorting
from typing import Any, Callable, Optional, Tuple

root = tk.Tk()
root.title("DermoSort")


def choosedir():
    global choosendir
    choosendir = filedialog.askdirectory()
    mylabel = Label(root, text=choosendir)
    mylabel.grid(row=0, columns=1)


def sort():
    sorting.main(rootfolder=choosendir)


button_exit = Button(root, text="Exit Program", command=root.quit)
button_exit.grid(row=1, column=2)

mylabel = Label(root, text="<choosendir>")
mylabel.grid(row=0, columns=1)

button_dir = Button(root, text="Choose Directory", command=choosedir)
button_dir.grid(row=0, column=2)
button_sort = Button(root, text="Sort Directory", command=sort)
button_sort.grid(row=1, column=1)

root.mainloop()
