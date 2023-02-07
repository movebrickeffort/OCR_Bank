import cv2
import numpy as np
from skimage import morphology
import skimage


def checkGray(chip):
    # chip_gray = cv2.cvtColor(chip,cv2.COLOR_BGR2GRAY)
    r, g, b = cv2.split(chip)
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    s_w, s_h = r.shape[:2]
    x = (r + b + g) / 3

    area_s = s_w * s_h
    # x = chip_gray
    r_gray = abs(r - x)
    g_gray = abs(g - x)
    b_gray = abs(b - x)
    r_sum = np.sum(r_gray) / area_s
    g_sum = np.sum(g_gray) / area_s
    b_sum = np.sum(b_gray) / area_s
    gray_degree = (r_sum + g_sum + b_sum) / 3
    if gray_degree < 10:
        return True
    else:
        return False


def Img_Outline(original_img):
    #original_img = cv2.imread(input_dir)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(gray_img, 165, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    #cv2.imshow('originImg',original_img)
    return original_img, gray_img, RedThresh, closed, opened

def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]          # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    print("angle",angle)
    box = np.int0(cv2.boxPoints(rect))
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    rows, cols = original_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (cols, rows))
    return result_img,draw_img

def correction(img):
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(img)
    result_img,draw_img = findContours_img(original_img,opened)
    return result_img,draw_img

def binary(img):
    ## 二值化
    # binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 1)
    ret, binary = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
    # ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def crop(img):
    #img = cv2.imread(input_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bin = binary(gray)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    (_, thresh) = cv2.threshold(gradient, 90, 255, cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    #box = np.int0(box)

    # draw a bounding box arounded the detected barcode and display the image
    #cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    print(box)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    print(Xs)
    print(Ys)
    x1 = min(Xs)+20
    x2 = max(Xs)-20
    y1 = min(Ys)+20
    y2 = max(Ys)-20
    hight = y2 - y1
    width = x2 - x1
    cropImg = img[y1:y1 + hight, x1:x1 + width]
    #cv2.imshow("Image", cropImg)
    #cropImg = binary(cropImg)

    #cv2.imshow("crop", cropImg)
    #cv2.imwrite("contoursImage2.jpg", img)

    return cropImg
def guassblur(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred

def noise(img):
    ## 去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(img, kernel, 2)
    cv2.imshow('erode', eroded)

    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(img, kernel, 2)
    cv2.imshow('dilated.png', dilated)

    #连通域
    # label = skimage.measure.label(img)
    # new_img = morphology.remove_small_objects(label, min_size=100, connectivity=2, in_place=False)
    # dilated = morphology.remove_small_objects(new_img, min_size=100, connectivity=2, in_place=False)
    # dilated = np.int8(dilated)
    # print(dilated)
    # cv2.imshow('dilated', dilated)

    return dilated


if __name__ == '__main__':
    #input_dir = 'correc/correction.jpg'
    #input_dir = 'correc/4.jpg'
    #input_dir = 'G:/SROIE/SROIE_test_images_task_3/X51006311714.jpg'
    #input_dir = 'D:/OCR/paddleOCR/PaddleOCR/doc/imgs/123.jpg'
    input_dir = 'data/03.png'
    img = cv2.imread(input_dir)
    img = guassblur(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   #  result_img, draw_img = correction(img)
   #  cv2.imshow('corr1', result_img)
   #  cv2.imwrite('corr1.png',result_img)
   #  cv2.imwrite('draw.png', draw_img)
   #  res = crop(img)
   #  cv2.imshow('crop1', res)
   #  cv2.imwrite('crop1.png',res)
   #  # cv2.imwrite('corr1.png',result_img)
   #  #     # cv2.imwrite('crop1.png',res)
   #  res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = binary(img)
    # cv2.imshow('res', res)
    cv2.imwrite('res20221130.png',res)
    #result_img,draw_img = correction(res)

    #print(result_img.shape)
    # result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    # res = crop(result_img)
    # cv2.imshow('crop2', res)
    # result_img, draw_img = correction(res)
    # cv2.imshow('corr2', result_img)

    #cropImg = crop(result_img)
    # ori = cv2.cvtColor(cropImg, cv2.COLOR_GRAY2BGR)
    # print(ori.shape)
    # cv2.imshow('ori',ori)


    #resImag = noise(cropImg)

    cv2.waitKey(0)