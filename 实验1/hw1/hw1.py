"""
姓名：张恩赐
学号：3019244118
"""
# img1[195][1] = 128,204,146
#------my-----------------
# img2[195][1] = 121,200,142
#------origin--------------
# MAP p[][]


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

RGBvalue = lambda img,x,y,channel:img[x][y][channel]
interpolation = lambda  leftPoint,rightPoint,P,leftValue,rightValue: \
                        (rightPoint - P)*leftValue +    \
                        (P - leftPoint )*rightValue

def judge(img1, img2, ratio):
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)

    diff = np.abs(img1 - img2)
    count = np.sum(diff > 1)

    assert count == 0, f'ratio={ratio}, Error!'
    print(f'ratio={ratio}, Success!')


def get_gt(img, ratio):
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    gt = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return gt

#A-------M--B
#|       P  |
#|          |
#C-------N--D
#
def resize_Biliner(img, ratio):
    ratioR = 1/ratio
    h = img.shape[0]
    w = img.shape[1]
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    ResizedImg = np.ndarray(shape=(new_h,new_w,3),dtype=np.uint8)
    for height in range(0,new_h):
        for width in range(0,new_w):
            p = (ratioR*height,ratioR*width)
            a = (math.floor(p[0]),math.floor(p[1]))
            b = (a[0],a[1]) if a[1] + 1 >= w else (a[0],a[1]+1) 
            c = (a[0],a[1]) if a[0] + 1 >= h else (a[0]+1,a[1])
            d = (c[0],b[1])
            M = (a[0],p[1])
            N = (c[0],p[1])
            for i in range(0,3):
                MValue = interpolation(a[1],b[1],p[1],RGBvalue(img,a[0],a[1],i),RGBvalue(img,b[0],b[1],i))
                NValue = interpolation(c[1],d[1],p[1],RGBvalue(img,c[0],c[1],i),RGBvalue(img,d[0],d[1],i))
                result = interpolation(M[0],N[0],p[0],MValue,NValue)
                ResizedImg[height][width][i] = result
    return ResizedImg
# to do
def resize_Nearest(img, ratio):
    """
    禁止使用cv2、torchvision等视觉库
    type img: ndarray(uint8)
    type ratio: float
    rtype: ndarray(uint8)
    """
    ratioR = 1/ratio
    h = img.shape[0]
    w = img.shape[1]
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    ResizedImg = np.ndarray(shape=(new_h,new_w,3),dtype=np.uint8)
    for height in range(0,new_h):
        for width in range(0,new_w):
            srcHeight = round(ratioR*height)
            srcWidth = round(ratioR*width)
            if(srcWidth > w - 1): srcWidth = w - 1
            if(srcHeight > h - 1): srcHeight = h - 1
            ResizedImg[height][width] = img[srcHeight][srcWidth]
    return ResizedImg

def show_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
#    ratios = [0.5, 0.8, 1.2, 1.5]
    ratios = [2.5]
    img = cv2.imread('images/img_1.jpeg')   # type(img) = ndarray, 一共有三张图片，都可以尝试

    start_time = time.time()
    for ratio in ratios:
        gt = get_gt(img, ratio)
        resized_img = resize_Biliner(img, ratio)
#        judge(gt, resized_img, ratio)

    end_time = time.time()
    total_time = end_time - start_time
#    show_images(img,resized_img)
    cv2.imwrite("img1_Biliner.jpg",resized_img,[cv2.IMWRITE_JPEG_QUALITY,100])
    print(f'用时{total_time:.4f}秒')
    print('Pass')
