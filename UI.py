import tkinter as tk
from tkinter.filedialog import askopenfile
from PIL import ImageTk, Image
from process import process
import os
from os import path

imgs = []
representation = {}
rep_icons = {}
texts = {}

def update_description(event, name, target):
    for widget in target.winfo_children():
        widget.destroy()
    tk.Label(target, image=representation[name]).pack()
    tk.Label(target, text=texts[name], wraplength=400).pack()
    try:
        target.last.configure(bg="white")
    except:
        pass
    event.widget.configure(bg="blue")
    target.last = event.widget

def on_click(event, boxs, image_path):
    x, y = event.x, event.y
    for box in boxs:
        if box.in_box(x, y):
            tp = tk.Toplevel(event.widget)
            img = Image.open(image_path)
            img = img.crop(box.rectangle)
            img = ImageTk.PhotoImage(image=img)
            imgs.append(img)
            tk.Label(tp, image=img).pack(side=tk.LEFT)

            frame = tk.Frame(tp)
            frame.pack(side=tk.LEFT)
            icons = tk.Frame(frame)
            icons.pack()
            description = tk.Frame(frame)
            description.pack()
            tk.Label(description, image=representation[box.clazz[0][0]]).pack()
            tk.Label(description, text=texts[box.clazz[0][0]], wraplength=400).pack()

            # TODO
            tk.Button(frame, text="保存实例").pack()

            for breed, possibility in box.clazz:
                icon = tk.Frame(icons)
                icon.pack(side=tk.LEFT)
                lab = tk.Label(icon, image=rep_icons[breed])
                lab.pack()
                lab.bind("<Button-1>", lambda event, a=breed, b=description: update_description(event, a, b))
                tk.Label(icon, text="{:.2f}%".format(possibility * 100)).pack()
            break


def show_photo():
    fin = askopenfile(title="choose a photo to process",
                      filetypes=[("JPG", ".jpg"), ("PNG", ".png")])
    path = fin.name
    fin.close()
    global imgs
    imgs.append(ImageTk.PhotoImage(file=path))

    img = imgs[-1]
    panel = tk.Toplevel()
    panel.title(path)
    canvas = tk.Canvas(panel, width=img.width() + 10, height=img.height() + 20)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=img)

    boxs = process(path)
    canvas.bind("<Button-1>", lambda event, bs=boxs, impath=path: on_click(event, bs, impath))
    for box in boxs:
        canvas.create_rectangle(box.rectangle, fill="", outline="red")
        bios = 0
        for t in box.clazz:
            string = "{0} {1}".format(*t)
            canvas.create_text(box.rectangle[0], box.rectangle[1] + bios, anchor="nw",
                               text=string, fill="red", font=("'Times New Roman'", 12))
            bios += 15


def main():
    root = tk.Tk()
    choose_file = tk.Button(master=root, text="选择图片", command=show_photo)
    choose_file.pack()
    tk.Button(root, text="保存的图片").pack()

    dir_ = path.join("data", "representation")
    files = os.listdir(dir_)
    for file in files:
        name = file[:-8].replace("_", " ")
        representation[name] = ImageTk.PhotoImage(file=path.join(dir_, file))
        rep_icons[name] = ImageTk.PhotoImage(image=Image.open(path.join(dir_, file)).resize((100, 100)))
        with open(path.join("data", "descriptions", "{}.txt".format(name)), encoding="utf-8") as fin:
            texts[name] = fin.read()
    tk.mainloop()


if __name__ == "__main__":
    main()
