import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

batch_size=256
epochs=15
epochs_2=12
learn_rate=0.001
input_dim=(32, 32, 3)
output_dim=10
test_size=0.1
valid_size=0.1/0.9
weight_decay = 1e-4

class ImageClassification():

  def __init__(self):
    '''self.x_train=[]
    self.x_valid=[]
    self.x_test=[]
    self.y_train=[]
    self.y_valid=[]
    self.y_test=[]'''
  def normalize(self,X_train,X_valid, X_test):
      #this function normalize inputs for zero mean and unit variance
      # it is used when training a model.
      # Input: training set and test set
      # Output: normalized training set and test set according to the trianing set statistics.
      mean = np.mean(X_train,axis=(0,1,2,3))
      std = np.std(X_train, axis=(0, 1, 2, 3))
      X_train = (X_train-mean)/(std+1e-7)
      X_valid = (X_valid-mean)/(std+1e-7)
      X_test = (X_test-mean)/(std+1e-7)
      return X_train, X_valid, X_test

  def create(self):
    (x_train, y_train), (x_test, y_test)=cifar10.load_data()
    x=np.concatenate((x_train, x_test))
    y=np.concatenate((y_train, y_test))
    x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=test_size, random_state=123)
    x_train, x_valid, y_train, y_valid=train_test_split(x_train, y_train, test_size=valid_size, random_state=123)

    #123
    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 
    x_valid = x_valid.astype('float32') 

    x_train, x_valid, x_test=self.normalize(x_train, x_valid, x_test)

    y_train=to_categorical(y_train,10)
    y_valid=to_categorical(y_valid,10)
    y_test=to_categorical(y_test,10)

    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

  def build(self):
    x_train, x_valid, x_test, y_train, y_valid, y_test=self.create()
    base_model=tensorflow.keras.applications.VGG16(
                        weights="imagenet",
                        input_shape=input_dim,
                        include_top=False,
                        classes=y_train.shape[1]
                    )
    #last = base_model.get_layer('block3_pool').output

    #base_model2 = Model(base_model.input, last)

    base_model.trainable=False

    model=Sequential() 
    model.add(base_model)
    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    #model.add(BatchNormalization())
    model.add(layers.Dense(512, activation='relu', kernel_constraint=unit_norm(), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(0.6))
    model.add(layers.Dense(256, activation='relu', kernel_constraint=unit_norm(), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_valid, y_valid),
                        verbose=1)

    base_model.trainable=True
    model.summary()
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    '''train_datagen = ImageDataGenerator(
                rescale=1. / 1,
                horizontal_flip=False)
    train_datagen.fit(x_train)
    train_generator = train_datagen.flow(x_train,
                                     y_train, 
                                     batch_size=batch_size)
    
    val_datagen = ImageDataGenerator(rescale=1. / 1,
        horizontal_flip=False)

    val_datagen.fit(x_valid)
    val_generator = val_datagen.flow(x_valid,
                                 y_valid,
                                 batch_size=batch_size)'''

    
    return model
  


if __name__=="__main__":
    IC=ImageClassification()
    x_train, x_valid, x_test, y_train, y_valid, y_test=IC.create()
    model=IC.build()

    train_steps_per_epoch = x_train.shape[0] // batch_size
    val_steps_per_epoch = x_valid.shape[0] // batch_size

    '''history = model.fit(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  epochs=epochs_2,
                                  verbose=1)'''

    history=model.fit(x_train, y_train,
                        epochs=epochs_2,
                        batch_size=batch_size,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=val_steps_per_epoch,
                        validation_data=(x_valid,y_valid),
                        verbose=1)


    score=model.evaluate(x_test, y_test, verbose=0)



    print('Test loss for model:', score[0])
    print('Test accuracy for model:', score[1])
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


