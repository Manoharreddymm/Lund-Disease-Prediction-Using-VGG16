#import required libraries
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Input,Flatten,Lambda
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

#the image size should be 224*224
Image_size=[224,224]

train_data=r'C:\Users\reddy\OneDrive\Desktop\LUNG_DISEASE\LUNG_DISEASE\LUNG_DISEASE\Train'
test_data=r'C:\Users\reddy\OneDrive\Desktop\LUNG_DISEASE\LUNG_DISEASE\LUNG_DISEASE\Test'


#adding preprocessing layer to the front and removing the last layer
vgg=VGG16(input_shape=Image_size+[3],weights='imagenet',include_top=False)

for layer in vgg.layers:
  layer.trainable=False


x=Flatten()(vgg.output)

prediction=Dense(4,activation='softmax')(x)

model=Model(inputs=vgg.input,outputs=prediction)



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_data=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_data=ImageDataGenerator(
    rescale=1./255
)

test_set = train_data.flow_from_directory(
    r'C:\Users\reddy\OneDrive\Desktop\LUNG_DISEASE\LUNG_DISEASE\LUNG_DISEASE\Test',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical'
)

train_set = train_data.flow_from_directory(
    r'C:\Users\reddy\OneDrive\Desktop\LUNG_DISEASE\LUNG_DISEASE\LUNG_DISEASE\Train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

tf.config.run_functions_eagerly(True)

model.fit(
    train_set,
    steps_per_epoch=748,
    epochs=1,
    validation_data=test_set,
    validation_steps=356
)


model.save(r'C:\Users\reddy\OneDrive\Desktop\project\model.h5')  # Save model to the specified path

print("Model saved successfully!")