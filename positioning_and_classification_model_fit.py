# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
import matplotlib.pyplot as plt
# matplotlib inline
from lxml import etree  # python的HTML/XML的解析器
import numpy as np
import glob

import matplotlib.patches as Rectangle
# print(tf.__version__)
# tf.test.is_gpu_available()


images = glob.glob("E:/Python/pythonProject_4/Image_target_positioning_and_classification/tmp/archive/images/*.jpg")
xmls = glob.glob("E:/Python/pythonProject_4/Image_target_positioning_and_classification/tmp/annotations/xmls/*.xml")
#获取文件的名称
names = [x.split("\\")[-1].split(".xml")[0] for x in xmls]
imgs_train = [img for img in images if img.split("\\")[-1].split(".jpg")[0] in names] # 标记的训练样本数3686 ，总样本数7000+
imgs_test = [img for img in images if img.split("\\")[-1].split(".jpg")[0] not in names]
#对其进行排序
imgs_train.sort(key=lambda x :x.split("\\")[-1].split(".jpg")[0]) # 按照第几种元素排序
xmls.sort(key=lambda x :x.split("\\")[-1].split(".xml")[0])
#排序后确定类别
names = [x.split("\\")[-1].split(".xml")[0] for x in xmls]
class_label = ["cat","dog"]
class_label_index = dict((name,index) for index,name in enumerate(class_label)) # enumerate :可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
label = [class_label_index[class_label[0]] if label.istitle() else class_label_index[class_label[1]] for label in names]
# 遍历names,if label.istitle return class_label_index[class_label[0]] else class_label_index[class_label[1]]

def to_labels(path):
    xml = open("{}".format(path)).read() # 以字符串格式打开一个path路径，并读取，
    sel = etree.HTML(xml) # 解析字符串格式的HTML文档对象，字符串格式变为_Element对象
    width = int(sel.xpath("//size/width/text()")[0])
    height = int(sel.xpath("//size/height/text()")[0])
    xmin = int(sel.xpath("//bndbox/xmin/text()")[0])
    xmax = int(sel.xpath("//bndbox/xmax/text()")[0])
    ymin = int(sel.xpath("//bndbox/ymin/text()")[0])
    ymax = int(sel.xpath("//bndbox/ymax/text()")[0])
    return [xmin/width,ymin/height,xmax/width,ymax/height]

#读取标注框位置信息
labels = [to_labels(path) for path in xmls]

out1,out2,out3,out4 = list(zip(*labels)) # 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
out1 = np.array(out1)
out2 = np.array(out2)
out3 = np.array(out3)
out4 = np.array(out4)
label = np.array(label)
label_datasets = tf.data.Dataset.from_tensor_slices((out1,out2,out3,out4,label))

def loda_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img,(224,224))
    img = img/127.5 - 1 #规划到-1到1之间
    return img
image_dataset = tf.data.Dataset.from_tensor_slices(imgs_train)
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_dataset = image_dataset.map(loda_image,num_parallel_calls=AUTOTUNE)
dataset = tf.data.Dataset.zip((image_dataset,label_datasets))

#%%设置训练数据和验证集数据的大小
test_count = int(len(imgs_train)*0.2)
train_count = len(imgs_train) - test_count
print(test_count,train_count)

dataset = dataset.shuffle(buffer_size=len(imgs_train))# 设置一个shuffle buffer size为imgs_train的一个缓冲区
train_dataset = dataset.skip(test_count) # 个人理解为从dataset数据集中选出test_count个对应数据以外的数据
test_dataset = dataset.take(test_count)

batch_size = 16
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
train_ds = train_dataset.shuffle(buffer_size=train_count).repeat().batch(batch_size) # 设置训练样本batch_size:取样间隔
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # 设置数据预取缓冲的元素数量为自动默认
test_ds = test_dataset.batch(batch_size) # 设置测试样本batch_size:取样间隔

from matplotlib.patches import Rectangle # 导入绘图标记框
for img,label in train_ds.take(1):
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    out1,out2,out3,out4,out5= label
    xmin,ymin,xmax,ymax = out1[0].numpy()*224,out2[0].numpy()*224,out3[0].numpy()*224,out4[0].numpy()*224 # 控制标记框缩放比例保持一致
    #给定左下角坐标
    rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color = "red",lw = 3)
    ax = plt.gca() # plt.plot()实际上会通过plt.gca()获得当前的Axes对象ax，然后再调用ax.plot()方法实现真正的绘图
    ax.axes.add_patch(rect)
    plt.title((class_label[out5[0]]).title())
    plt.show()

# 创建模型
xcpetion = tf.keras.applications.Xception(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

inputs = tf.keras.layers.Input(shape=(224,224,3))
x = xcpetion(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x1 = tf.keras.layers.Dense(2048, activation='relu')(x)

x1 = tf.keras.layers.Dense(256, activation='relu')(x1)
out1 = tf.keras.layers.Dense(1,name="out1")(x1)
out2 = tf.keras.layers.Dense(1,name="out2")(x1)
out3 = tf.keras.layers.Dense(1,name="out3")(x1)
out4 = tf.keras.layers.Dense(1,name="out4")(x1)

x2 = tf.keras.layers.Dense(1024, activation='relu')(x)
x2 = tf.keras.layers.Dense(256, activation='relu')(x2)
out_class = tf.keras.layers.Dense(1,activation='sigmoid',name='out_item')(x2)

prediction = [out1,out2,out3,out4,out_class]
model = tf.keras.models.Model(inputs=inputs,outputs=prediction)
model.summary()

# 模型参数编译
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss={'out1':'mse',
                    'out2':'mse',
                    'out3':'mse',
                    'out4':'mse',
                    'out_item':'binary_crossentropy'},
              metrics=["mae","acc"])
steps_per_eooch = train_count//batch_size
validation_steps = test_count//batch_size

history = model.fit(train_ds,epochs=2,steps_per_epoch=steps_per_eooch,validation_data=test_ds,validation_steps=validation_steps)

# 模型估计
model.save("detect_v1.h5")
new_model = tf.keras.models.load_model("E:/Python/pythonProject_4/Image_target_positioning_and_classification/detect_v1.h5")
# plt.figure(figsize=(8, 24))
# for img, _ in test_ds.skip(8).take(1):
#     out1, out2, out3, out4, out5 = new_model.predict(img)
#     for i in range(6):
#         plt.subplot(6, 1, i + 1)
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))
#         xmin, ymin, xmax, ymax = out1[i] * 224, out2[i] * 224, out3[i] * 224, out4[i] * 224
#         # 给定左下角坐标
#         rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color="red", lw = 3)
#         ax = plt.gca()
#         ax.axes.add_patch(rect)
#         result_label = class_label[round(float(out5[i]))]
#         ax.axes.text(xmin-0.5, ymin-4, '{label_}'.format(label_ = result_label), size='x-large', color='white', bbox={'facecolor':'red', 'alpha':1, 'edgecolor':'none', 'pad':1}) # alpha : 透明度
#         # plt.title((class_label[round(float(out5[i]))]).title())
#     plt.show()