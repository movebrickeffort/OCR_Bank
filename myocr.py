from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

ocr = PaddleOCR(use_angle_cls=True, gpu_use = True, gpu_mem=1000)

img_path = 'noise_img.png'
output_path = 'out1.txt'

result = ocr.ocr(img_path, cls=True)
print(result)
print(result[0][0][1][1])
sum = 0
for i in range(len(result[0])):
    sum = sum + result[0][i][1][1]

print(sum/len(result[0]))


# for root,dirs,files in os.walk('C:/Users/Administrator/Desktop/card'): #使用API快速把某目录下的路径分类
#
#     for file in files: #用for循环把文件打印出来
#         print(os.path.join(root,file))
#         result = ocr.ocr(os.path.join(root,file), cls=True)
#         print(len(result[0]))
#         with open(output_path, 'a', encoding='utf-8') as f:
#             f.write(file)
#             f.write('\n')
#
#             for i in range(len(result[0])):
#                 print(result[0][i][1][0])
#
#                 f.write(result[0][i][1][0])
#                 f.write('\n')
#             f.write('\n')


