import cv2

# image = cv2.imread('D:/OCR/bank-receipt-recognition/test1/test.jpg')
image = cv2.imread('test1/11.png')
print(image)
imgry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 先转化为灰度图
cv2.imwrite('test1/imgry.png',imgry)
binary = cv2.adaptiveThreshold(imgry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 1)
cv2.imwrite('test1/binary1.png',binary)

# 也可以先滤波 然后在二值

# # 腐蚀
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# eroded = cv2.erode(binary,kernel,10)
# cv2.imwrite('erode.png',eroded)
#
# # 膨胀
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# dilated = cv2.dilate(eroded, kernel, 10)
# cv2.imwrite('dilated.png',dilated)