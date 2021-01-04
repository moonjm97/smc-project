###
#cifar dataset :
#60000 32x32 color images in 10classes(each 6000)
# training images: 50000 | test images: 10000
# training: total 50000 images, divided into 5 batches, each with 10000 images
# test: total 10000 images in one batch (1000 randomly-selected images from each class)

#vgg16
###

import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from keras.utils.np_utils import to_categorical
# 숫자 -> One-hot Vector 를 위한 라이브러리
#from keras.datasets import cifar10


# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  # NOTE: <=0.4 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)


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

# 배열에서 1차원, 2차원과 같은 개념을 텐서에서는 'rank'라고 표현
# rank가 0인 경우(Scalar): 1 | rank가 1인 경우(Vector): [1,2,3] | rank가 2인 경우(Matrix): [[1,2,3], [4,5,6]]
# axis는 rank의
# tf.squeeze(input, axis=None, name=None)
# -> rank가 1인 차원 찾아 제거하여 스칼라값으로 바꾼다
#    []를 제거해서 하나로 만들어준다.
# ex. [[0],[1],[2]] -> [0,1,2]

# tf.one_hot(indices(텐서), depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)

y_train_onehot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_onehot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

inputs = Input(shape=(32,32,3), dtype='float32')

# convolutional layer
# feature detect
# filter = kernel
# *filter 는 가중치를 의미한다.
# Conv2D:
# tf.keras.layers.Conv2D(filters, filter_size, strides=(1,1), padding='valid',
#                       activation=None, kernel_initializer='glorot_uniform')
# filter 적용한 후 padding='same' | make input size,output size same
#                padding= 'valid' | 유효한 영역만 출력, 따라서 output size < inputsize
# 64,128,512-특징 filter의 개수(64개의 특징 filters), (3,3)-단면에서 움직이는 filter size)
# Activation Function: feature map이 추출된 이후에 map의 값을 "있다,없다" 와 같은 비선형 값으로 바꿔줌

conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1-1')(inputs)
conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1-2')(conv1_1)
maxpooling1 = MaxPooling2D((2, 2), padding='same', name='maxpooling1')(conv1_2) #(16,16,64)

conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2-1')(maxpooling1)
conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2-2')(conv2_1)
maxpooling2 = MaxPooling2D((2, 2), padding='same', name='maxpooling2')(conv2_2) #(8,8,128)

conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3-1')(maxpooling2)
conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3-2')(conv3_1)
conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3-3')(conv3_2)
maxpooling3 = MaxPooling2D((2, 2), padding='same', name='maxpooling3')(conv3_3) #(4,4,256)

conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4-1')(maxpooling3)
conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4-2')(conv4_1)
conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4-3')(conv4_2)
maxpooling4 = MaxPooling2D((2, 2), padding='same', name='maxpooling4')(conv4_3) #(2,2,512)

conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5-1')(maxpooling4)
conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5-2')(conv5_1)
conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5-3')(conv5_2)
maxpooling5 = MaxPooling2D((2, 2), padding='same', name='maxpooling5')(conv5_3) #(1,1,512)

# Flatten: 2차원 데이터를 1차원 데이터로 why? 완전연결시키기위해?
#  con2D로 Feature Map 만들고->MaxPooling 으로 차원 감소->Dense Layer 사용하여
#  감소된 차원의 Feature Map 들만 Input 과 Output 과 완전 연결 계층 생성하여 효율적인 학습 가능
# (?) Dense layer: Input 과 Output 연결
fc1 = Flatten(name='fc1')(maxpooling5)
# fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
# fc2 = Dense(2048, activation='relu', name='fc2')(fc1)
# 4096->2048->1042->... 얼마나 한번에 압축시키느냐의 차이 -> 너무 구체적이어서 overfitting 발생 가능
# Dense512 vs Dense 256 무슨 차이? -> 깊이가 깊어질수록 overfitting 발생 가능
fc2 = Dense(256, activation='relu', name='fc2')(fc1)
outputs = Dense(10, activation='relu', name='outputs')(fc2)

# 마지막 단계에서 어떤 함수를 쓰던 accuracy 같은 성능 면이 달라지는 것은 아니지만
# softmax 는 결과값을 확률값으로 바꿔주어서 보기 편하게 만드는 역할을 한다.

# create a model
# param
vgg16 = Model(inputs, outputs, name='vgg16')
vgg16.summary()

# compile the model
# 모델 학습과정 설정 | 손실 함수 및 최적화 방법
# optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # lower learning rate, better performance
vgg16.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# train the model
# 모델 학습
vgg16.fit(x_train, y_train_onehot, epochs=30)

# evaluate the model with test set
# 모델 평가하기
vgg16.evaluate(x_test, y_test_onehot, verbose=2)

# predict the model with test set
# 모델 사용하기
# vgg16.predict(x) : x라는 input을 넣어서 output 예측 생성

# loss: 3.0565 - accuracy: 0.3367


