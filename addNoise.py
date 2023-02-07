import numpy as np
import cv2
#读取图片
img = cv2.imread("data/03.png")
img_height,img_width,img_channels = img.shape
#设置高斯分布的均值和方差
mean = 0
#设置高斯分布的标准差
sigma = 35
#根据均值和标准差生成符合高斯分布的噪声
gauss = np.random.normal(mean,sigma,(img_height,img_width,img_channels))
#给图片添加高斯噪声
noise_img = img + gauss
#设置图片添加高斯噪声之后的像素值的范围
noise_img = np.clip(noise_img,a_min=0,a_max=255)
#保存图片
cv2.imwrite("noise_img.png",noise_img)