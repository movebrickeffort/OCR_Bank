import os
from paddleocr import PaddleOCR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def resort(a):
    index = 0
    res = []
    while (index < len(a)):
        x = a[index][0][-1][1]
        tmp = []
        j = index
        while (j < len(a) and abs(x - a[j][0][-1][1]) <= 10):
            tmp.append(a[j][0][0][0])
            j = j + 1
        # 列表从index到j找到小的
        tmp.sort()
        print("tmp", tmp)
        i = 0
        for i in range(len(tmp)):
            for k in range(index, j):
                if (a[k][0][0][0] == tmp[i]):
                    res.append(a[k])
        index = j

    return res

model = PaddleOCR(use_angle_cls=False, cls_model_dir='D:/OCR/paddleOCR/PaddleOCR\inference/ch_ppocr_mobile_v2.0_cls_infer/',det_model_dir = 'D:/OCR/paddleOCR/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer/',rec_model_dir = 'D:/OCR/paddleOCR2/PaddleOCR/inference/ch_ppocr_server_v2.0_rec_infer/')  # gpu_mem=1000 use_gpu=False use_space_char=False

# img_path = '../SROIE_DATA/0325updated.task2train(626p)/'
img_path = 'D:/OCR/paddleOCR/PaddleOCR/doc/imgs/zhipiaotest/'
img_list = os.listdir(img_path)
img_list = [img for img in img_list if img[-3:] == 'jpg']

for img in img_list:
    with open('D:/OCR/paddleOCR/PaddleOCR/doc/imgs/zhipiaotxttest/' + img[:-3] + 'txt', 'w',encoding='utf-8') as f:
        result = model.ocr(img_path + img, cls=True)
        print(result)
        result = resort(result)
        print(result)
        for line in result:
            ans = []
            print(line[1][0])
            # ans = [str(int(l)) for li in line[0] for l in li]
            # print(ans)
            f.write(line[1][0])
            f.write('\n')
    # except:
    #     print(img)
    #     continue

