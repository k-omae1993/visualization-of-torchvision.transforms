import numpy as np

def change(img):
    img = np.asarray(img)
    img = np.transpose(img, (1,2,0))
    img = np.zeros(img)
    return img

