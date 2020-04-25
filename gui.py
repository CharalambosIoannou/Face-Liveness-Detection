import tkinter as tk
from tkinter import filedialog
from liveness_from_video import run
from liveness_from_photos import single_image
from tkinter import messagebox
from tkinter.ttk import Separator
# base code for tkinter obtain from https://stackoverflow.com/questions/45257305/how-to-divide-container-into-different-sets-of-columns-in-tkinter-gui/45257560#45257560


def UploadAction(event=None):
	filename = filedialog.askopenfilename()
	print('Selected:', filename)
	# filename_chosen = filename
	global filename_chosen
	filename_chosen = filename
	entryText.set(filename_chosen)
	return filename
	
def get_file():
	try:
		file_to_read = filename_chosen
		print("f: " , file_to_read)
	except NameError:
		messagebox.showinfo("Invalid Input", "Please select an image file")
		return
	if file_to_read.endswith('.jpg') or file_to_read.endswith('.png'):
		file_to_read = file_to_read.replace("\\", "\\\\")
		print("new: " , file_to_read)
		r = single_image(file_to_read)
		if (r == None):
			messagebox.showinfo("No Face Detected", "The program could not locate a face in the image")
		print("r",r)
	else:
		messagebox.showinfo("Invalid File Chosen", "Please choose a .jpg or .png file")
		
def real_time():
	res = run()
	if (res == "no face"):
		messagebox.showinfo("Error", "No face detected")
	elif res =="multiple faces":
		messagebox.showinfo("Error", "Multiple faces detected")

	
root = tk.Tk()
root.title('Facial Liveness Detection System')
root.geometry('{}x{}'.format(900, 500))


topFrame = tk.Frame(root, bg = 'lavender', width = 900, height=100, relief = 'raised')
buttons = tk.Frame(root)
lbl1 = tk.Label(root,text='\n\n\nPlease select an option:', width = 0, height = 0, padx = 0, pady = 0,bg="lavender",font=("Arial", 15))
lbl = tk.Label(root,text='Facial Liveness Detection', width = 15, height = 0, padx = 20, pady = 2,bg="lavender",font=("Arial Bold", 20))

lbl1.grid(row = 0, column = 0, sticky='we')
lbl.grid(row = 0, column = 0, sticky='we')
topFrame.grid(row = 0, column = 0, columnspan = 3,  sticky="w")


labelCps2 = tk.Label(root, text="\n\n\n\nChoose an image file (.jpg or .png) by pressing the Open Button below", width = 0, height = 0, padx = 10, pady = 0)
labelCps = tk.Label(root, text="Real-time Liveness Detection", width = 0, height = 0, padx = 10, pady = 10,font=("Arial Bold", 10))
labelIgn = tk.Label(root, text='Input Image Liveness Detection', width = 0, height = 0, padx = 10, pady = 10,font=("Arial Bold", 10))


labelCps.grid(row = 1, column = 0, sticky='we')
labelIgn.grid(row = 1, column = 2, sticky='we')
labelCps2.grid(row = 1, column = 2, sticky='we')


labelCps = tk.Label(root, text="Pressing the Run Liveness \n button  will open the device's camera", width = 0, height = 0, padx = 10, pady = 10)


labelCps.grid(row = 2, column = 0, sticky='we')
# labelCps1.grid(row = 3, column = 2, sticky='we')




cpsFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100

ignFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100
ignFrame.grid(row = 2, column = 2)



entryText = tk.StringVar()
labelIgn = tk.Label(ignFrame, text = 'Selected image:')
button = tk.Button(ignFrame, text='Open', command=UploadAction)
entryIgn = tk.Entry(ignFrame, state='disabled', textvariable=entryText )
labelCps1 = tk.Label(root, text="Pressing the Validate Image \n button  will make a prediction \n according to the image selected above ", width = 0, height = 0, padx = 0, pady = 0)

labelIgn.grid(row = 0, column = 0)
entryIgn.grid(row = 0, column = 1)
button.grid(row = 0, column = 2)
labelCps1.grid(row = 3, column = 2)





root.grid_rowconfigure(4, pad = 50)

applyButton = tk.Button(root, text = 'Run Liveness', padx = 30, pady = 15,command=real_time)
applyButton.grid(row = 4, columnspan = 1)

applyButton1 = tk.Button(root, text = 'Validate Image', padx = 30, pady = 15, command=get_file )
applyButton1.grid(row = 4,column=2,  columnspan = 1)


root.grid_rowconfigure(5, pad = 50)


dataButton = tk.Button(root ,text = 'Quit',anchor = 'center', padx = 15, pady = 15,command = root.quit,bg="lavender")
# dataButton.grid(row = 5, column = 1, padx=100)
dataButton.place(relx=0.475, rely=0.93, anchor='se')
# dataButton.place(relx=0.95, rely=0.9, anchor='se')

sep = Separator(root, orient='vertical')
sep.grid(column=1, row=1, rowspan=4,padx=20, sticky='ns')


root.mainloop()




