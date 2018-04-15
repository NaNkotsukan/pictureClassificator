from PIL import Image
import numpy as np
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D, normalization
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D,MaxPooling2D

class Predict:
    def m(self):
        model=Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(normalization.BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(normalization.BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(normalization.BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(normalization.BatchNormalization())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Activation('softmax'))
        return model

    def __init__(self):
        self.model=self.m()
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        self.model.summary()
        self.model.load_weights("model/param0.hdf5")

    def __call__(self,image):
        try:
            img = Image.open(image)
            size=min((img.width,img.height))
            img.crop((0, 0, size, size)).resize((100, 100), Image.LANCZOS)
            x=np.asarray(img.crop((0, 0, size, size)).resize((100, 100), Image.LANCZOS),dtype="float32").reshape((1,100,100,3))
            if x.shape==(100,100):x=np.concatenate((x.reshape(100,100,1),x.reshape(100,100,1),x.reshape(100,100,1)),axis=2)
            elif x.shape==(100,100,4):x=x[:, :, 0:3]
            return self.model.predict(x)[0]
        except:
            return Exception("err")


if __name__=='__main__':
    predict=Predict()
    res=predict("imagePass")
    classes=("イラスト","写真")
    txt=str(res.max()*100)+"%の確率で"+classes[res.argmax()]+"です。"  
    print(txt)