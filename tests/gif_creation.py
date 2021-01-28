"""Script to test gif creation and to tweak the parameters
"""
import os
import cv2
from utils.gif import save_frames_as_gif


if __name__=="__main__":
    save_dir = "saves"
    if not os.path.exists(save_dir):
        print(f"{save_dir} was not found, exiting")
        exit(1)
    folders = os.listdir(save_dir)
    if len(folders) < 1:
        print(f"{save_dir} is empty, exiting")
        exit(1)
    folder = sorted(folders)[0]
    folder = os.path.join(save_dir, folder)
    image_files = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.splitext(file)[-1] == ".png"
    ]
    image_files = sorted(image_files)
    if len(image_files) < 5:
        print("not enough image files were found, exiting")
        exit(1)
    save_frames_as_gif(image_files, path=folder)
    