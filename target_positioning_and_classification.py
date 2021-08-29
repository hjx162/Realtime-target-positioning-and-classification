# -*- coding=GBK -*-
import cv2 as cv
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # �����ͼ��ǿ�

new_model = tf.keras.models.load_model("E:/Python/pythonProject_4/Image_target_positioning_and_classification/detect_v1.h5") # ����cat,dogʶ��ģ��

# outPutDirName='E:\Python\pythonProject_4\Image_target_positioning_and_classification\dog_1.mp4' # or dog.mp4
frameFrequency=1

class_label = ["cat","dog"]
# ������ͷ��ȡͼƬ
def video_demo():
    times = 0
    capture = cv.VideoCapture(0)  # ������ͷ��0��������豸id������ж������ͷ����������������ֵ
    while True:
        times += 1
        res, image = capture.read()  # ��ȡ����ͷ,���ܷ���������������һ��������bool�͵�ret����ֵΪTrue��False��������û�ж���ͼƬ���ڶ���������frame���ǵ�ǰ��ȡһ֡��ͼƬ
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            image = cv.flip(image, 1)  # flip():ͼ��ת����   �ڶ������� С��0: 180����ת������0: ���µߵ�������0: ˮƽ�ߵ�(����ͼ)

            image_1 = image.astype("float32")  # ��������ת�� : ������ͼƬ��int8����ת��Ϊfloat32����
            image_1 = image_1 / 255
            image_1 = cv.resize(image_1, (224, 224))
            image_2 = np.expand_dims(image_1, axis=0)
            image_2 = tf.convert_to_tensor(image_2)

            out1, out2, out3, out4, out5 = new_model.predict(image_2)  # ʶ��ģ�ͽ���cat,dogʶ��

            pred_mask = model.predict(image_2)
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]

            pred_mask_1 = tf.keras.preprocessing.image.array_to_img(pred_mask[0])  # PILת
            pred_mask_2 = np.asarray(pred_mask_1)
            # depth_image = cv.applyColorMap(pred_mask_2, cv.COLORMAP_VIRIDIS)
            # depth_image = depth_image.astype("float32")
            # depth_image = depth_image / 255

            h, w = pred_mask_2.shape
            pixel_255 = []
            for m in range(h):
                for n in range(w):
                    if pred_mask_2[n, m] == 255:
                        pixel_255.append([n, m])
            pixel_255 = np.array(pixel_255)
            xmin = min(pixel_255[:, 1])
            ymin = min(pixel_255[:, 0])
            xmax = max(pixel_255[:, 1])
            ymax = max(pixel_255[:, 0])

            result_label = class_label[round(float(out5))]
            cv.rectangle(image_1, (xmin - 10, ymin), (xmax, ymax), (0, 0, 255), 2)  # ������������һ������ɫ��һ���Ǳ߿���
            cv.putText(image_1, '{label_}'.format(label_=result_label), (xmin - 10, ymin - 3), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            cv.imshow("seg_video", image_1)

            if cv.waitKey(10) & 0xFF == ord('q'):  # �����ע
                 break


video_demo()
cv.destroyAllWindows()