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
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', name = 'dense1_'+ str(n))(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name = 'dense2_'+ str(n))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax', name = 'dense3_'+ str(n))(x)
    model = Model(input=base_model.input, output=predictions)
    model.load_weights('model/Resnet50_channel1.hdf5')

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    v_o = 0
    # v_o = 181
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

def build_model(k):
    #view 1
    model_original1 = res50_model(0)
    model_texture1 = res50_model(1)
    model_shape1 = res50_model(2)
    merged1 = concatenate([model_original1.output, model_texture1.output, model_shape1.output], axis=-1)
    merged1 = Dense(2, activation='softmax', name='KBC_v1')(merged1)
    print 'view1'
    # view 2
    model_original2 = res50_model(3)
    model_texture2 = res50_model(4)
    model_shape2 = res50_model(5)
    merged2 = concatenate([model_original2.output, model_texture2.output, model_shape2.output], axis=-1)
    merged2 = Dense(2, activation='softmax', name='KBC_v2')(merged2)
    print 'view2'
    # view 3
    model_original3 = res50_model(6)
    model_texture3 = res50_model(7)
    model_shape3 = res50_model(8)
    merged3 = concatenate([model_original3.output, model_texture3.output, model_shape3.output], axis=-1)
    merged3 = Dense(2, activation='softmax', name='KBC_v3')(merged3)
    print 'view3'
    # view 4
    model_original4 = res50_model(9)
    model_texture4 = res50_model(10)
    model_shape4 = res50_model(11)
    merged4 = concatenate([model_original4.output, model_texture4.output, model_shape4.output], axis=-1)
    merged4 = Dense(2, activation='softmax', name='KBC_v4')(merged4)
    print 'view4'
    # view 5
    model_original5 = res50_model(12)
    model_texture5 = res50_model(13)
    model_shape5 = res50_model(14)
    merged5 = concatenate([model_original5.output, model_texture5.output, model_shape5.output], axis=-1)
    merged5 = Dense(2, activation='softmax', name='KBC_v5')(merged5)
    print 'view5'
    # view 6
    model_original6 = res50_model(15)
    model_texture6 = res50_model(16)
    model_shape6 = res50_model(17)
    merged6 = concatenate([model_original6.output, model_texture6.output, model_shape6.output], axis=-1)
    merged6 = Dense(2, activation='softmax', name='KBC_v6')(merged6)
    print 'view6'
    # view 7
    model_original7 = res50_model(18)
    model_texture7 = res50_model(19)
    model_shape7 = res50_model(20)
    merged7 = concatenate([model_original7.output, model_texture7.output, model_shape7.output], axis=-1)
    merged7 = Dense(2, activation='softmax', name='KBC_v7')(merged7)
    print 'view7'
    # view 8
    model_original8 = res50_model(21)
    model_texture8 = res50_model(22)
    model_shape8 = res50_model(23)
    merged8 = concatenate([model_original8.output, model_texture8.output, model_shape8.output], axis=-1)
    merged8 = Dense(2, activation='softmax', name='KBC_v8')(merged8)
    print 'view8'
    # view 9
    model_original9 = res50_model(24)
    model_texture9 = res50_model(25)
    model_shape9 = res50_model(26)
    merged9 = concatenate([model_original9.output, model_texture9.output, model_shape9.output], axis=-1)
    merged9 = Dense(2, activation='softmax', name='KBC_v9')(merged9)
    print 'view9'

    merged = concatenate([merged1,merged2,merged3,merged4,merged5,merged6,merged7,merged8,merged9], axis=-1)
    final = Dense(1, activation='sigmoid',name='MVKBC')(merged)
    model_merged = Model(input=[model_original1.input, model_texture1.input, model_shape1.input,
                         model_original2.input, model_texture2.input, model_shape2.input,
                         model_original3.input, model_texture3.input, model_shape3.input,
                         model_original4.input, model_texture4.input, model_shape4.input,
                         model_original5.input, model_texture5.input, model_shape5.input,
                         model_original6.input, model_texture6.input, model_shape6.input,
                         model_original7.input, model_texture7.input, model_shape7.input,
                         model_original8.input, model_texture8.input, model_shape8.input,
                         model_original9.input, model_texture9.input, model_shape9.input], output=final)

    # model_merged.load_weights('model/KBC_view1_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view2_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view3_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view4_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view5_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view6_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view7_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view8_best_' + str(k) + '.hdf5', by_name=True)
    # model_merged.load_weights('model/KBC_view9_best_' + str(k) + '.hdf5', by_name=True)

    return model_merged

def cla_evaluate(YY_a, pred_a, score_a):
    acc = metrics.accuracy_score(YY_a, pred_a)
    auc = metrics.roc_auc_score(YY_a,score_a)
    CM = metrics.confusion_matrix(YY_a, pred_a)
    sens=float(CM[1,1])/float(CM[1,1]+CM[1,0])
    spec=float(CM[0,0])/float(CM[0,0]+CM[0,1])
    return acc, auc, sens, spec

for k in range(1,11):    #cross-validation

    # build multi-view KBC model
    model_merged = build_model(k)
    model_merged = multi_gpu_model(model_merged, gpus=8)
    # model_merged = multi_gpu_model(model_merged, gpus=2)
    model_merged.compile(loss=penality_loss, optimizer=SGD(lr=0.0001, momentum=0.9,decay = 1e-4),metrics=['accuracy'])

    # data
    with tf.device('/cpu:0'):
        [train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12, train13,
         train14, train15, train16, train17, train18, train19, train20, train21, train22, train23, train24, train25,
         train26, train27, test1, test2, test3,test4, test5, test6, test7, test8, test9, test10, test11, test12, test13,
         test14, test15, test16, test17, test18,test19, test20, test21, test22, test23, test24, test25, test26,test27,
         train_labels, test_labels] = generate_data(k,img_width,img_height,channels)

    nb_epoch = 100
    nb_batch = 16

    json_string = model_merged.to_json()
    open('model/MVKBC_architecture.json', 'w').write(json_string)

    best_model = ModelCheckpoint(filepath=("model/MVKBC_best_" + str(k) + ".hdf5"),
                                 monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto')

    model_merged.fit([train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12, train13,
                      train14, train15, train16, train17, train18, train19, train20, train21, train22, train23, train24, train25,
                      train26, train27], train_labels, batch_size=nb_batch, epochs=nb_epoch, shuffle=True, verbose=1,validation_split=0.1,callbacks=[best_model])

    model_merged.save_weights('model/MVKBC_last_' + str(k) + '.h5', overwrite=True)

    # testing
    model_merged.load_weights('model/MVKBC_best_' + str(k) + '.hdf5')
    score_test_pre = model_merged.predict([test1, test2, test3,test4, test5, test6, test7, test8, test9, test10, test11, test12, test13,
                                           test14, test15, test16, test17, test18,test19, test20, test21, test22, test23, test24, test25, test26,test27])
    pred_test_a = np.round(score_test_pre)
    test_acc, test_auc, test_sens, test_spec = cla_evaluate(test_labels, pred_test_a, score_test_pre)

    line_test = "test:acc=%f,auc=%f, sens=%f,spec=%f \n" % (test_acc, test_auc, test_sens, test_spec)
    print line_test





