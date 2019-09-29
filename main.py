import tkinter as tk
from tkinter import *
import random
import time
import datetime
#import subprocess
import webbrowser


def fun():
    master = tk.Tk()

    # Reproduced under license from United States
    # Department of Agriculture.
    # Download a copy from from <URL: http://phaseit.net/tmp/gif/s1.gif>.
    image_file = "media\logo0.gif"
    image = tk.PhotoImage(file = image_file)

        # Fill the screen.
    screen_width  = master.winfo_screenwidth()
    screen_height = master.winfo_screenheight()
    s = tk.Canvas(master, width  = screen_width,
                   height = screen_height)

        # Remove window-manager decorations:  don't allow
    # the user to destroy or iconify the splash screen.
    master.overrideredirect(1)

    # Center the image.
    s.create_image(screen_width / 2, screen_height / 2,
                   image = image)
    s.create_text( screen_width / 2, screen_height / 2,
          text = "石油信息平台",
          fill = "magenta2", font = "normal 45")
    s.pack()

    # After five seconds--5000 milliseconds--remove the
    # splash screen.
    master.after(1000, lambda: master.destroy())
    master.mainloop()

fun()

# import apps\01basic\tkinter_basic
import tkinter_basic
tkinter_basic.root.mainloop()
# import tkinter_advance
# tkinter_advance.root.mainloop()
# import gui
# gui.root.mainloop()