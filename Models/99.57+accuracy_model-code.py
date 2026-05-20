#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 21:02:04 2025

@author: amit
"""

from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet_v2 import MobileNetV2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications.xception import Xception
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU

from tensorflow.keras.applications import EfficientNetB3

SIZE = 128
train_images = []
train_labels = [] 
for directory_path in glob.glob("/home/amit/data_sets/PBC_dataset_normal_DIB_224/*"):
    label = directory_path.split("/")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)






x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test = (train_images, test_images, train_labels,test_labels)

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

label_encoder = LabelEncoder()

# Fit and transform the y_train labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)


"""rabbin dataset_study """

SIZE = 112
train_images = []
train_labels = [] 
for directory_path in glob.glob("/home/amit/data_sets/rabbin dataset/Train/*"):
    label = directory_path.split("/")[-1]
    #label = directory_path.split("/")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)



test_images = []
test_labels = [] 
for directory_path in glob.glob("/home/amit/data_sets/rabbin dataset/TestA/*"):
    label = directory_path.split("/")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        # img = enhance_image(img)
        # img = remove_hair(img) 
        test_images.append(img)
        test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


#x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = (train_images, test_images, train_labels,test_labels)

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# label_encoder = LabelEncoder()

# Fit and transform the y_train labels
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.fit_transform(y_test)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)



















from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Lambda, Reshape, Multiply
import tensorflow as tf

def sk_block2(input, filters, strides=1):
    # Branch 1
    branch1 = Conv2D(filters, (3, 3), strides=strides, padding="same")(input)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU()(branch1)
    branch1 = Conv2D(filters, (3,3), strides=1, padding="same")(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU()(branch1)

    # Branch 2
    branch2 = Conv2D(filters, (5,5), strides=strides, padding="same", dilation_rate=3)(input)
    branch2 = BatchNormalization()(branch2)
    branch2 = ReLU()(branch2)
    branch3 = Conv2D(filters, (5,5), strides=1, padding="same", dilation_rate=5)(branch2)
    branch3 = BatchNormalization()(branch3)
    branch3 = ReLU()(branch3)
    branch4 = Add()([branch2, branch3])

    fused = Add()([branch1, branch3])

    attention = GlobalAveragePooling2D()(fused)
    x_a = Dense(filters // 16, activation='relu')(attention)
    x_a = Dense(2 * filters, activation='sigmoid')(x_a)

    def split_attention(x):
        return tf.split(x, num_or_size_splits=2, axis=1)
    
    def split_attention_output_shape(input_shape):
        filters = input_shape[-1] // 2
        return [input_shape[:-1] + (filters,), input_shape[:-1] + (filters,)]

    a1, a2 = Lambda(split_attention, output_shape=split_attention_output_shape)(x_a)
    a1 = Reshape((1, 1, filters))(a1)
    a2 = Reshape((1, 1, filters))(a2)

    output = Add()([Multiply()([branch1, a1]), Multiply()([branch4, a2])])
    return output



def multiScale_feature_fusion_block(sampled_dense_output1,sampled_dense_output2,sampled_dense_output3,sampled_AcsConv_layer):
  concat_layer = tf.keras.layers.Concatenate()([sampled_dense_output1,sampled_dense_output2,sampled_dense_output3,sampled_AcsConv_layer])
  convolution_layer = tf.keras.layers.Conv2D(128,1,padding="same")(concat_layer)
  x1 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output1])
  x2 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output2])
  x3 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output3])
  x4 = tf.keras.layers.Concatenate()([convolution_layer,sampled_AcsConv_layer])
  return [x1,x2,x3,x4]




def CAB_Block(feature_map):
  x1 = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(feature_map)
  x1 = tf.keras.activations.gelu(x1, approximate=False)

  x3 = tf.keras.layers.DepthwiseConv2D(5, dilation_rate=3,padding="same")(x1)

  x3 = tf.keras.layers.DepthwiseConv2D(7,dilation_rate = 5,padding="same")(x3)

  Att = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(x3)

  XL = tf.keras.layers.multiply([x1,Att])

  XL = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(XL)

  XL = tf.keras.activations.gelu(XL, approximate=False)

  X_final = tf.keras.layers.Add()([XL, feature_map])

  x2 = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(X_final)

  x2 = tf.keras.layers.DepthwiseConv2D(3,padding="same")(x2)

  x1 = tf.keras.activations.gelu(x1, approximate=False)

  x2 = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(x2)

  final_output = tf.keras.layers.Add()([x2,X_final])

  return final_output



input_shape = (128,128,3)
input_tensor = Input(shape = input_shape)
#base_model = Xception(weights ='imagenet',include_top = False,input_tensor= input_tensor)
# base_model = DenseNet201(weights ='imagenet',include_top = False,input_tensor= input_tensor)
base_model = MobileNetV2(weights ='imagenet',include_top = False,input_tensor= input_tensor)

# Make all the layers in model_2_base_model trainable
base_model.trainable = False



base_model.summary()
feature1 = base_model.get_layer('block_3_expand').output
feature2 = base_model.get_layer('block_6_expand').output
feature3 = base_model.get_layer('block_13_expand').output

# feature1 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(feature1)

print( feature1.shape)
print(feature2.shape)
print(feature3.shape)




# output of the AcsConv block
# AcsConv_layer = AcsConv_block(feature3)

# AcsConv_layer1 = sk_block1(feature3,64)
# AcsConv_layer.shape

newfeature1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(feature1)
print(newfeature1.shape)
# newfeature2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature2)

#newAcsConv_layer = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(AcsConv_layer)

AcsConv_layer4=sk_block2(feature3,64)
newfeature3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature3)

AcsConv_layer41 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(AcsConv_layer4)

# print(AcsConv_layer41.shape)
print(AcsConv_layer41.shape)
print(newfeature3.shape)
print(newfeature1.shape)
print(feature2.shape)








x1, x2, x3 , x4 = multiScale_feature_fusion_block(newfeature1,newfeature3,feature2,
AcsConv_layer41)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)







#multiscalce_feature_fusion_list = multiScale_feature_fusion_block(newfeature1,feature2,newfeature3,newAcsConv_layer)
#applying Context attention block
cs1 = tf.keras.layers.Conv2D(128,1,padding="same")(x1)
print(cs1.shape)
cs1 = CAB_Block(cs1)
cs2 = tf.keras.layers.Conv2D(128,1,padding="same")(x2)
cs2 = CAB_Block(cs2)
cs3 = tf.keras.layers.Conv2D(128,1,padding="same")(x3)
cs3 = CAB_Block(cs3)
cs4 = tf.keras.layers.Conv2D(128,1,padding="same")(x4)
cs4 = CAB_Block(cs4)


print(cs1.shape)
print(cs2.shape)
print(cs3.shape)
print(cs4.shape)




final_output=tf.keras.layers.Concatenate()([cs1,cs2,cs3,cs4])
print(final_output.shape)




final_pooled_output = tf.keras.layers.GlobalAveragePooling2D()(final_output)

flattened_output = tf.keras.layers.Dense(256,activation="relu")(final_pooled_output)
final_dense_layer = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(flattened_output)

model = Model(inputs=input_tensor, outputs=final_dense_layer)

model.summary()


model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])






datagen = ImageDataGenerator(
       rotation_range=20,
       zoom_range=0.1,
       horizontal_flip=True,
       vertical_flip=True,
      )
datagen.fit(x_train)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
checkpoint_path = "/home/amit/Reviewer_answers/dilation_rate_variation/DKCAB(3,5)_CAB(5,7)con1.keras" 

# Create ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True, 
    monitor='val_accuracy',
    save_freq="epoch",
    verbose=1
)

# Add Early Stopping to prevent overfitting

history = model.fit(
    datagen.flow(x_train, y_train_one_hot, batch_size=32),
    epochs=50,
    validation_data=(x_test, y_test_one_hot),
    callbacks=[checkpoint_callback],verbose=1  # All callbacks
)

from sklearn.metrics import cohen_kappa_score
import numpy as np

# Load best weights if you saved them
model.load_weights(checkpoint_path)

# Predict on test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Calculate Quadratic Weighted Kappa
qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
print("Quadratic Weighted Kappa (QWK):", qwk)






