import scipy.io as sio
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
import sys

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


img_width, img_height = 64, 64
channels = 1
smooth = 1.

#Define the neural network
def get_unet():
    inputs = Input((img_width, img_height,channels))
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], axis=-1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=-1)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], axis=-1)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], axis=-1)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9,decay = 1e-4), loss='binary_crossentropy', metrics=[dice_coef])

    return model


for k in range(1,11):
    # label
    data_all = h5py.File('segmentation_data/Nodule_' + str(k)+'.mat')

    # data
    train_data = [data_all[element[0]][:] for element in data_all['train']]
    test_data = [data_all[element[0]][:] for element in data_all['test']]
    mask_train = [data_all[element[0]][:] for element in data_all['mask_train']]
    mask_test = [data_all[element[0]][:] for element in data_all['mask_test']]

    num_train = len(train_data)
    train = np.empty((num_train,img_width,img_height,channels),dtype="double")
    for i in range(num_train):
            train[i,:,:,0] = train_data[i].T

    train_mask = np.empty((num_train,img_width,img_height,channels),dtype="double")
    for i in range(num_train):
            train_mask[i,:,:,0] = mask_train[i].T

    num_test = len(test_data)
    test = np.empty((num_test, img_width, img_height, channels), dtype="double")
    for j in range(num_test):
        test[j,:,:,0] = test_data[j].T

    test_mask = np.empty((num_test, img_width, img_height, channels), dtype="double")
    for j in range(num_test):
        test_mask[j,:,:,0] = mask_test[j].T


    #the U-net model
    model = get_unet()
    print "Check: final output of the network:"
    print model.output_shape
    json_string = model.to_json()
    open('seg_model/architecture.json', 'w').write(json_string)

    #============  Training ==================================
    checkpointer = ModelCheckpoint(filepath='seg_model/training_best_weights'+str(k)+'.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

    #training settings
    N_epochs = 100
    batch_size = 32
    model.fit(train, train_mask, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])
    #========== Save and test the last model ===================
    model.save_weights('seg_model/training_last_weights'+str(k)+'.h5', overwrite=True)

    model.load_weights('seg_model/training_best_weights'+str(k)+'.h5')
    model.predict(test)

