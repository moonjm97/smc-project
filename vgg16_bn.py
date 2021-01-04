import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D, BatchNormalization, Activation
# 숫자 -> One-hot Vector 를 위한 라이브러리
#from keras.datasets import cifar10


# Fix the gpu memory issue
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # NOTE: <=0.3 with GTX 1050 (2GB)
#session = tf.compat.v1.Session(config=config)

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
y_train_onehot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_onehot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

inputs = Input(shape=(32,32,3), dtype='float32')

# convolutional layer
# feature detect
# filter = kernel
# Conv2D:
# tf.keras.layers.Conv2D(filters, filter_size, strides=(1,1), padding='valid',
#                       activation=None, kernel_initializer='glorot_uniform')
# filter 적용한 후 padding='same' (make input size,output size same)
# 64,128,512-특징 filter의 개수(64개의 특징 filters), (3,3)-단면에서 움직이는 filter size)
# Activation Function: feature map이 추출된 이후에 map의 값을 "있다,없다" 와 같은 비선형 값으로 바꿔줌

# BatchNormalization
# 학습 시 미니배치 단위로 정규화
# 데이터분포가 평균이 0, 분산이 1이 되도록 정규화
# gradient vanishing or exploding 막기 위해 각 층에서의 활성화값이 적당히 분포되도록 조정
# -> 어떤 특징에 너무 치우치지 않도록
# advantages : 학습 속도 개선 / 초깃값에 크게 의존하지 않는다 / over-fitting 억제

# axis=0 행 방향으로 동작 -> 작업결과가 행으로 나타남

# axis=1 열 방향으로 동작 -> 작업결과가 열로 나타남
# axis = 1 은 두 번째로 높은 차원을 기준으로 합치는 것이다.
# 2차원 자료라면 -> 1차원을 기준으로 붙이고
# 3차원 자료라면 -> 2차원을 기준으로 붙이면 된다.

# block 1
conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1-1')(inputs)
conv1_1bn = BatchNormalization(axis=1, name='conv1_1bn')(conv1_1)
conv1_1re = Activation('relu', name='conv1_1re')(conv1_1bn)

conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1-2')(conv1_1re)
conv1_2bn = BatchNormalization(axis=1, name='conv1_2bn')(conv1_2)
conv1_2re = Activation('relu', name='conv1_2re')(conv1_2bn)
maxpooling1 = MaxPooling2D((2, 2), 2, padding='same', name='maxpooling1')(conv1_2re) #(16,16,64)

# block 2
conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2-1')(maxpooling1)
conv2_1bn = BatchNormalization(axis=1, name='conv2_1bn')(conv2_1)
conv2_1re = Activation('relu', name='conv2_1re')(conv2_1bn)

conv2_2 = Conv2D(128,ㅋ (3, 3), activation='relu', padding='same', name='conv2-2')(conv2_1re)
conv2_2bn = BatchNormalization(axis=1, name='conv2_2bn')(conv2_2)
conv2_2re = Activation('relu', name='conv2_2re')(conv2_2bn)
maxpooling2 = MaxPooling2D((2, 2), 2, padding='same', name='maxpooling2')(conv2_1re) #(8,8,128)

# block 3
conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3-1')(maxpooling2)
conv3_1bn = BatchNormalization(axis=1, name='conv3_1bn')(conv3_1)
conv3_1re = Activation('relu', name='conv3_1re')(conv3_1bn)

conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3-1')(conv3_1re)
conv3_2bn = BatchNormalization(axis=1, name='conv3_2bn')(conv3_2)
conv3_2re = Activation('relu', name='conv3_2re')(conv3_2bn)

conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3-1')(conv3_2re)
conv3_3bn = BatchNormalization(axis=1, name='conv3_3bn')(conv3_3)
conv3_3re = Activation('relu', name='conv3_3re')(conv3_3bn)
maxpooling3 = MaxPooling2D((2, 2), 2, padding='same', name='maxpooling3')(conv3_3re) #(4,4,256)

# block 4
conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4-1')(maxpooling3)
conv4_1bn = BatchNormalization(axis=1, name='conv4_1bn')(conv4_1)
conv4_1re = Activation('relu', name='conv4_1re')(conv4_1bn)

conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4-2')(conv4_1re)
conv4_2bn = BatchNormalization(axis=1, name='conv4_2bn')(conv4_2)
conv4_2re = Activation('relu', name='conv4_2re')(conv4_2bn)

conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4-3')(conv4_2re)
conv4_3bn = BatchNormalization(axis=1, name='conv4_3bn')(conv4_3)
conv4_3re = Activation('relu', name='conv4_3re')(conv4_3bn)
maxpooling4 = MaxPooling2D((2, 2), 2, padding='same', name='maxpooling4')(conv4_3re) #(2,2,512)

# block 5
conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5-1')(maxpooling2)
conv5_1bn = BatchNormalization(axis=1, name='conv5_1bn')(conv5_1)
conv5_1re = Activation('relu', name='conv5_1re')(conv5_1bn)

conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5-2')(conv5_1re)
conv5_2bn = BatchNormalization(axis=1, name='conv5_2bn')(conv5_2)
conv5_2re = Activation('relu', name='conv5_2re')(conv5_2bn)

conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5-3')(conv5_2re)
conv5_3bn = BatchNormalization(axis=1, name='conv5_3bn')(conv5_3)
conv5_3re = Activation('relu', name='conv5_3re')(conv5_3bn)
maxpooling5 = MaxPooling2D((2, 2), 2, padding='same', name='maxpooling3')(conv5_3re) #(1,1,512)

fc1 = Flatten(name='fc1')(maxpooling5)
fc2 = Dense(256, activation='relu', name='fc2')(fc1)
outputs = Dense(10, activation='relu', name='outputs')(fc2)


# create a model
# param
vgg16bn = Model(inputs, outputs, name='vgg16bn')
vgg16bn.summary()

# compile the model
# 모델 학습과정 설정 | 손실 함수 및 최적화 방법
# optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # lower learning rate, better performance
vgg16bn.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# train the model
# 모델 학습
vgg16bn.fit(x_train, y_train_onehot, epochs=30)

# evaluate the model with test set
# 모델 평가하기
vgg16bn.evaluate(x_test, y_test_onehot, verbose=2)

# lr: 0.001 loss: 4.5617 - accuracy: 0.2793
# lr: 0.0005 loss: 3.0825 - accuracy: 0.3949
# lr: 0.00025