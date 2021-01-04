###
#cifar dataset :
#60000 32x32 color images in 10classes(each 6000)
# training images: 50000 | test images: 10000
# training: total 50000 images, divided into 5 batches, each with 10000 images
# test: total 10000 images in one batch (1000 randomly-selected images from each class)

#ResNet-50
# model의 layer가 깊어질수록 오히려 성능 떨어짐.
# gradient vanishing/exploding 문제 해결 위해 고안
# ResNet은 skip connection을 이용한 residual learning을 통해 layer가 깊어짐에 따른 gradient vanishing 문제를 해결
###

import tensorflow as tf
from tensorflow.keras.layers import Add, ZeroPadding2D, Conv2D, Activation, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras import Input, Model

# gpu 메모리 문제 해결
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



# 데이터셋 생성
# x=data y=label
# label은 어떤 역할? feature = input, label = output (class 번호)
# ex) feature = 털, 뾰족한 귀 | ouput = 고양이
cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# convert pixel range from 0(black)-255(white) to 0-1
# float32 실수
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# one-hot-encoding
y_train_onehot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_onehot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

inputs = Input(shape=(32,32,3), dtype='float32')

# convolutional layer
# Conv2D:
# tf.keras.layers.Conv2D(filters, filter_size, strides=(1,1), padding='valid',
#                       activation=None, kernel_initializer='glorot_uniform')
# tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None, **kwargs)
# strides -> downsampling

# block1
x = ZeroPadding2D(padding=(3, 3))(inputs)
x = Conv2D(64, (7, 7), strides=2, padding='valid',activation=None)(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1,1))(x) # (16,16,64)
# tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,**kwargs)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x) # (8,8,64)
shortcut = Conv2D(256,(1,1))(x) # (8,8,256)

# block 2
x = Conv2D(64,(1,1),strides=1,padding='valid')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(64,(3,3),strides=1,padding='same')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(256,(1,1),strides=1,padding='valid')(x)
x = Conv2D(256,(1,1),strides=1,padding='valid')(shortcut)
x = BatchNormalization(axis=1)(x)
shortcut = BatchNormalization(axis=1)(shortcut)
add_1 = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu')(add_1) # (8,8,256)
shortcut = x

# block 3

x = Conv2D(64,(1,1),strides=1,padding='valid')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(64,(3,3),strides=1,padding='same')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(256,(1,1),strides=1,padding='valid')(x)
x = BatchNormalization(axis=1)(x)
add_2 = Add()([x,shortcut])
x = Activation('relu')(add_2) # (8,8,256)
shortcut = x

# block 4

x = Conv2D(64,(1,1),strides=1,padding='valid')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(64,(3,3),strides=1,padding='same')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(256,(1,1),strides=1,padding='valid')(x)
x = BatchNormalization(axis=1)(x)
add_3 = Add()([x,shortcut])
x = Activation('relu')(add_3)
shortcut = Conv2D(512,(1,1))(x)

# block 5
x = Conv2D(128,(1,1),strides=2,padding='valid')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(128,(3,3),strides=1,padding='same')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)

x = Conv2D(512,(1,1),strides=1,padding='valid')(x)
x = Conv2D(512,(1,1),strides=1,padding='valid')(shortcut)
x = BatchNormalization(axis=1)(x)
shortcut = BatchNormalization(axis=1)(shortcut)
add_4 = Add()([x,shortcut])
x = Activation('relu')(add_4) # (4,4,512)

x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation='softmax')(x)

resnet50 = Model(inputs, outputs)
resnet50.summary()

# compile the model
# 모델 학습과정 설정 | 손실 함수 및 최적화 방법
# optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # lower learning rate, better performance
resnet50.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# train the model
# 모델 학습
resnet50.fit(x_train, y_train_onehot, epochs=40)

# evaluate the model with test set
# 모델 평가하기
resnet50.evaluate(x_test, y_test_onehot, verbose=2)

# lr: 0.0001  loss: 0.7362 - accuracy: 0.7607
# lr: 0.00005 loss: 0.7421 - accuracy: 0.7477
# lr: 0.001   loss: 0.9238 - accuracy: 0.7767








