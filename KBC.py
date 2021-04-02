import scipy.io as sio
import h5py
import numpy as np
import tensorflow as tf
from numpy import *
# import math
from keras.applications.resnet50 import ResNet50
from generate_train_test import generate_data
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn import metrics
from keras.utils import multi_gpu_model

img_width, img_height, channels = 224, 224, 1

def res50_model(n):
    inputs = Input((img_width, img_height, channels))
    base_model = ResNet50(input_tensor = inputs, input_shape = (img_width, img_height, channels), weights=None, include_top=False, k=n)
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', name = 'dense1_'+ str(n))(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name = 'dense2_'+ str(n))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax', name = 'dense3_'+ str(n))(x)
    model = Model(input=base_model.input, output=predictions)
    model.load_weights('model/Resnet50_channel1.hdf5')

    v_o = 0
    for layer in model.layers[:v_o]:
        layer.trainable = False

    return model

# generate a loss function (penality+cross-entropy)
def penality_loss(y_true, y_pred):
    sub_val = y_true - y_pred
    a = K.abs(sub_val) <= 0.5 * K.ones_like(sub_val)
    a = K.cast(a, dtype='float')
    b = sub_val > 0.5 * K.ones_like(sub_val)
    b = K.cast(b, dtype='float')
    c = sub_val <= -0.5 * K.ones_like(sub_val)
    c = K.cast(c, dtype='float')
    d = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    cost = a * 1 * d + b * 2 * d + c * 1 * d
    return cost

def build_model(f):
    #view 1
    model_original1 = res50_model(0)
    model_texture1 = res50_model(1)
    model_shape1 = res50_model(2)
    merged1 = concatenate([model_original1.output, model_texture1.output, model_shape1.output], axis=-1)
    print 'view1'

    final = Dense(1, activation='sigmoid', name='KBC_'+f)(merged1)
    model_merged = Model(input=[model_original1.input, model_texture1.input, model_shape1.input], output=final)

    return model_merged

def cla_evaluate(YY_a, pred_a, score_a):
    acc = metrics.accuracy_score(YY_a, pred_a)
    auc = metrics.roc_auc_score(YY_a,score_a)
    CM = metrics.confusion_matrix(YY_a, pred_a)
    sens=float(CM[1,1])/float(CM[1,1]+CM[1,0])
    spec=float(CM[0,0])/float(CM[0,0]+CM[0,1])
    return acc, auc, sens, spec

for k in range(1,11):    #cross-validation
    f = 'view1'
    #build KBC model for view 1
    model_merged = build_model(f)
    model_merged = multi_gpu_model(model_merged, gpus=2)
    model_merged.compile(loss=penality_loss, optimizer=SGD(lr=0.0001, momentum=0.9,decay = 1e-4),metrics=['accuracy'])
    # data
    with tf.device('/cpu:0'):
        [train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12, train13,
         train14, train15, train16, train17, train18, train19, train20, train21, train22, train23, train24, train25,
         train26, train27, test1, test2, test3,test4, test5, test6, test7, test8, test9, test10, test11, test12, test13,
         test14, test15, test16, test17, test18,test19, test20, test21, test22, test23, test24, test25, test26,test27,
         train_labels, test_labels] = generate_data(k,img_width,img_height,channels)

    nb_epoch = 100
    nb_batch = 32

    json_string = model_merged.to_json()
    open('model/KBC_architecture.json', 'w').write(json_string)

    best_model = ModelCheckpoint(filepath=('model/KBC_'+f+'_best_' + str(k) + '.hdf5'),
                                 monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto')

    model_merged.fit([train1, train2, train3], train_labels, batch_size=nb_batch, epochs=nb_epoch, shuffle=True, verbose=1,
                     validation_split=0.1,callbacks=[best_model])

    model_merged.save_weights('model/KBC_'+f+'_last_' + str(k) + '.h5', overwrite=True)

    # testing
    model_merged.load_weights('model/KBC_'+f+'_best_' + str(k) + '.hdf5')
    score_test_pre = model_merged.predict([test1, test2, test3])
    pred_test_a = np.round(score_test_pre)
    test_acc, test_auc, test_sens, test_spec = cla_evaluate(test_labels, pred_test_a, score_test_pre)

    line_test = "test:acc=%f,auc=%f, sens=%f,spec=%f \n" % (test_acc, test_auc, test_sens, test_spec)
    print line_test







