import argparse
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Normalize
from change import change

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser for visualization")
    parser.add_argument('--image_path', type=str, default='image/test.jpg', help='path to image_path')
    parser.add_argument('--Resize', type=int, default=0, help='path to Resize')
    parser.add_argument('--Normalize', type=int, default=0, help='path to Normalize')
    parser.add_argument('--CenterCrop', type=int, default=0, help='path to CenterCrop')
    parser.add_argument('--Grayscale', type=int, default=0, help='path to Grayscale')
    parser.add_argument('--RandomAffine', type=int, default=0, help='path to RandomAffine')
    parser.add_argument('--RandomCrop', type=int, default=0, help='path to RandomCrop')
    parser.add_argument('--RandomGrayscale', type=int, default=0, help='path to RandomGrayscale')
    parser.add_argument('--RandomHorizontalFlip', type=int, default=0, help='path to RandomHorizontalFlip')
    args = parser.parse_args()


    img = Image.open(args.image_path)
    transform_totensor = transforms.ToTensor()
    before_img = transform_totensor(img)
    pre_img = transform_totensor(img)
    #Resize
    if args.Resize == 0:
        pre_img = pre_img
    else:
        transform_resize = transforms.Resize(224)
        pre_img = transform_resize(pre_img)

    #Normalize
    if args.Normalize == 0:
        pre_img = pre_img
    else:
        transform_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        pre_img = transform_normalize(pre_img)
    
    #CenterCrop
    if args.CenterCrop == 0:
        pre_img = pre_img
    else:
        transform_centercrop = transforms.CenterCrop(150)
        pre_img = transform_centercrop(pre_img)

    #Grayscale
    if args.Grayscale == 0:
        pre_img = pre_img
    else:
        transform_grayscale = transforms.Grayscale(3)
        pre_img = transform_grayscale(pre_img)

    #RandomAffine
    if args.RandomAffine == 0:
        pre_img = pre_img
    else:
        transform_randomaffine = transforms.RandomAffine(45)
        pre_img = transform_randomaffine(pre_img)

    #RandomCrop
    if args.RandomCrop == 0:
        pre_img = pre_img
    else:
        transform_randomcrop = transforms.RandomCrop(100)
        pre_img = transform_randomcrop(pre_img)

    #RandomGrayscale
    if args.RandomGrayscale == 0:
        pre_img = pre_img
    else:
        transform_randomgrayscale = transforms.RandomGrayscale(0.5)
        pre_img = transform_randomgrayscale(pre_img)

    #RandomHorizontalFlip
    if args.RandomHorizontalFlip == 0:
        pre_img = pre_img
    else:
        transform_randomhorizontalflip = transforms.RandomHorizontalFlip()
        pre_img = transform_randomhorizontalflip(pre_img)

    #tensorからnumpyに変換
    before_img = change(before_img)
    pre_img = change(pre_img)

    #matplotlibで変更前と変更後の画像の可視化
    #fig1 = plt.figure(figsize=(640,480))
    #ax1 = fig1.add_subplot(111)
    #ax1.set_title('before preprocessing')
    #mg1 = ax1.imshow(before_img)

    #fig2 = plt.figure(figsize=(640,480))
    #ax2 = fig2.add_subplot(111)
    #ax2.set_title('after prerprocessing')
    #img2 = ax2.imshow(pre_img)
    
    
    plt.subplot(121).imshow(before_img)
    plt.subplot(122).imshow(pre_img)
    plt.show()