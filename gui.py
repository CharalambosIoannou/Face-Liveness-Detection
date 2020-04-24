import tkinter as tk
from tkinter import filedialog
from liveness_from_video import run
from liveness_from_photos import single_image
from tkinter import messagebox

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
	file_to_read = file_to_read.replace("\\", "\\\\")
	print("new: " , file_to_read)
	single_image(file_to_read)
		
	
root = tk.Tk()
root.title('Steady State Data Processing')
root.geometry('{}x{}'.format(900, 500))


topFrame = tk.Frame(root, bg = 'lavender', width = 900, height=100, relief = 'raised')
lbl1 = tk.Label(root,text='\n\n\nPlease select an option:', width = 0, height = 0, padx = 0, pady = 0,bg="lavender",font=("Arial", 15))
lbl = tk.Label(root,text='Facial Liveness Detection', width = 15, height = 0, padx = 20, pady = 2,bg="lavender",font=("Arial Bold", 20))

lbl1.grid(row = 0, column = 0, sticky='we')
lbl.grid(row = 0, column = 0, sticky='we')
topFrame.grid(row = 0, column = 0, columnspan = 3,  sticky="w")


labelCps2 = tk.Label(root, text="\n\n\n\n\nChoose an image file by pressing the Open Button below", width = 0, height = 0, padx = 10, pady = 0)
labelCps = tk.Label(root, text="Real-time Liveness Detection", width = 0, height = 0, padx = 10, pady = 0,font=("Arial Bold", 10))
labelIgn = tk.Label(root, text='Input Image Liveness Detection', width = 0, height = 0, padx = 10, pady = 0,font=("Arial Bold", 10))



labelCps.grid(row = 1, column = 0, sticky='we')
labelIgn.grid(row = 1, column = 1, sticky='we')
labelCps2.grid(row = 1, column = 1, sticky='we')

labelCps = tk.Label(root, text="Pressing the Run Liveness \n button  will open the device's camera", width = 0, height = 0, padx = 10, pady = 10)


labelCps.grid(row = 2, column = 0, sticky='we')




cpsFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100

ignFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100
ignFrame.grid(row = 2, column = 1,  sticky="nsew")



entryText = tk.StringVar()
labelCps1 = tk.Label(ignFrame, text="\n\n\n\n\nPressing the Validate Image \n button  will make a prediction \n according to the image selected above ", width = 0, height = 0, padx = 10, pady = 0)
labelIgn = tk.Label(ignFrame, justify = 'left', text = 'Selected image:')
button = tk.Button(ignFrame, text='Open', command=UploadAction)
entryIgn = tk.Entry(ignFrame, state='disabled', textvariable=entryText )

labelIgn.grid(row = 0, column = 0, sticky = 'w')
entryIgn.grid(row = 0, column = 1)
labelCps1.grid(row = 0, column = 1, sticky='we')


button.grid(row = 0, column = 2)






root.grid_rowconfigure(2, pad = 50)

applyButton = tk.Button(root, text = 'Run Liveness', padx = 30, pady = 15,command=run)
applyButton.grid(row = 3, columnspan = 1)

applyButton1 = tk.Button(root, text = 'Validate Image', padx = 30, pady = 15, command=get_file )
applyButton1.grid(row = 3, column=1,columnspan = 1)

root.mainloop()




