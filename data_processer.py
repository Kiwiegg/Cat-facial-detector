import os
import numpy as np
import pandas as pd

from PIL import Image
# from PIL import ImageDraw

img_size = 224

# modify this for different data files
path = "D:\Cat dataset\data\CAT_06"

dirs = sorted(os.listdir(path))

data = {
    'imgs': [],
    'lmks': [],
    'faces': []      
}

# resizes the image to 224x224, by first resizing the longer side to 
# 224 pixel while maintaning ratio, then pasting the resized image
# onto a 224x224 black screen
def resize_img (img):
    old_size = img.size
    ratio = float(img_size) / max(old_size)
    new_size = (old_size[0] * ratio, old_size[1] * ratio)
    img = img.resize((int(new_size[0]), int(new_size[1])))
    dw = int((img_size - new_size[0]) / 2)
    dh = int((img_size - new_size[1]) / 2)

    new_img = Image.new("RGB", (img_size, img_size))
    new_img.paste(img, (dw, dh))
    
    return new_img, ratio, dw, dh

count  = 0
for f in dirs:
    if '.cat' not in f:
        continue
    
    # read the landmarks
    dataframe = pd.read_csv(path + "\\" + f, sep = " ", header=None)
    landmarks = dataframe.to_numpy()
    
    if (landmarks[0][0] != 9):
        continue

    #delete the first and last element, then resize it into to columns
    landmarks = np.delete(landmarks, -1, 1)
    landmarks = np.delete(landmarks, 0, 1)
    landmarks = np.reshape(landmarks, (-1, 2))
    
    # read the image
    image_path = path + "\\" + f
    image_path = image_path[:-4]
    img = Image.open(image_path)
    
    # resize the image 
    img, ratio, dw, dh = resize_img(img)
    landmarks = ((landmarks * ratio) + np.array([dw, dh])).astype(np.int)
    face = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])
    
    img_data = np.asarray(img)

    data["imgs"].append(img_data)
    data["lmks"].append(landmarks.flatten())
    data["faces"].append(face.flatten())
    
    """
    draw = ImageDraw.Draw(img, mode="RGB")
    for l in landmarks:
        img =  draw.point(l, fill=(128, 0, 128))
    img.save("test", format="png")
    break  
    
    """
    
    if count % 20 == 0:
        print(count)
    count += 1

# modify this for different data files 
np.save("processed_data06.npy", np.array(data))
    
