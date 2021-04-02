import scipy.io as sio
import h5py
import numpy as np
from keras.utils import np_utils
# import math
def data_g(train,test,img_width, img_height, channels):
    num_train = len(train)
    train1 = np.empty((num_train, img_width, img_height, channels), dtype="double")
    for i in range(num_train):
        train1[i, :, :, 0] = train[i].T
        # train1[i, :, :, 1] = train[i].T
        # train1[i, :, :, 2] = train[i].T

    num_test = len(test)
    test1 = np.empty((num_test, img_width, img_height, channels), dtype="double")
    for j in range(num_test):
        test1[j, :, :, 0] = test[j].T
        # test1[j, :, :, 1] = test[j].T
        # test1[j, :, :, 2] = test[j].T
    return train1,test1

def generate_data(k,img_width1,img_height1,channels):
    nam = 'data/LIDC_nodule_d3_all_9_agu_224'
    # label
    data_all1 = h5py.File(nam + '/nodule_' + str(k) + '_1_original.mat')
    data_all2 = h5py.File(nam + '/nodule_' + str(k) + '_1_texture.mat')
    data_all3 = h5py.File(nam + '/nodule_' + str(k) + '_1_shape.mat')

    data_all4 = h5py.File(nam + '/nodule_' + str(k) + '_2_original.mat')
    data_all5 = h5py.File(nam + '/nodule_' + str(k) + '_2_texture.mat')
    data_all6 = h5py.File(nam + '/nodule_' + str(k) + '_2_shape.mat')

    data_all7 = h5py.File(nam + '/nodule_' + str(k) + '_3_original.mat')
    data_all8 = h5py.File(nam + '/nodule_' + str(k) + '_3_texture.mat')
    data_all9 = h5py.File(nam + '/nodule_' + str(k) + '_3_shape.mat')

    data_all10 = h5py.File(nam + '/nodule_' + str(k) + '_4_original.mat')
    data_all11 = h5py.File(nam + '/nodule_' + str(k) + '_4_texture.mat')
    data_all12 = h5py.File(nam + '/nodule_' + str(k) + '_4_shape.mat')

    data_all13 = h5py.File(nam + '/nodule_' + str(k) + '_5_original.mat')
    data_all14 = h5py.File(nam + '/nodule_' + str(k) + '_5_texture.mat')
    data_all15 = h5py.File(nam + '/nodule_' + str(k) + '_5_shape.mat')

    data_all16 = h5py.File(nam + '/nodule_' + str(k) + '_6_original.mat')
    data_all17 = h5py.File(nam + '/nodule_' + str(k) + '_6_texture.mat')
    data_all18 = h5py.File(nam + '/nodule_' + str(k) + '_6_shape.mat')

    data_all19 = h5py.File(nam + '/nodule_' + str(k) + '_7_original.mat')
    data_all20 = h5py.File(nam + '/nodule_' + str(k) + '_7_texture.mat')
    data_all21 = h5py.File(nam + '/nodule_' + str(k) + '_7_shape.mat')

    data_all22 = h5py.File(nam + '/nodule_' + str(k) + '_8_original.mat')
    data_all23 = h5py.File(nam + '/nodule_' + str(k) + '_8_texture.mat')
    data_all24 = h5py.File(nam + '/nodule_' + str(k) + '_8_shape.mat')

    data_all25 = h5py.File(nam + '/nodule_' + str(k) + '_9_original.mat')
    data_all26 = h5py.File(nam + '/nodule_' + str(k) + '_9_texture.mat')
    data_all27 = h5py.File(nam + '/nodule_' + str(k) + '_9_shape.mat')

    train_labels = data_all1['labels_tr'][:].T
    test_labels = data_all1['labels_te'][:].T

    # train_labels = np_utils.to_categorical(train_labels, 2)
    # test_labels = np_utils.to_categorical(test_labels, 2)

    # view1(OTS)
    train_data = [data_all1[element[0]][:] for element in data_all1['train']]
    test_data = [data_all1[element[0]][:] for element in data_all1['test']]
    train1, test1 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all2[element[0]][:] for element in data_all2['train']]
    test_data = [data_all2[element[0]][:] for element in data_all2['test']]
    train2, test2 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all3[element[0]][:] for element in data_all3['train']]
    test_data = [data_all3[element[0]][:] for element in data_all3['test']]
    train3, test3 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view2(OTS)
    train_data = [data_all4[element[0]][:] for element in data_all4['train']]
    test_data = [data_all4[element[0]][:] for element in data_all4['test']]
    train4, test4 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all5[element[0]][:] for element in data_all5['train']]
    test_data = [data_all5[element[0]][:] for element in data_all5['test']]
    train5, test5 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all6[element[0]][:] for element in data_all6['train']]
    test_data = [data_all6[element[0]][:] for element in data_all6['test']]
    train6, test6 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view3(OTS)
    train_data = [data_all7[element[0]][:] for element in data_all7['train']]
    test_data = [data_all7[element[0]][:] for element in data_all7['test']]
    train7, test7 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all8[element[0]][:] for element in data_all8['train']]
    test_data = [data_all8[element[0]][:] for element in data_all8['test']]
    train8, test8 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all9[element[0]][:] for element in data_all9['train']]
    test_data = [data_all9[element[0]][:] for element in data_all9['test']]
    train9, test9 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view4(OTS)
    train_data = [data_all10[element[0]][:] for element in data_all10['train']]
    test_data = [data_all10[element[0]][:] for element in data_all10['test']]
    train10, test10 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all11[element[0]][:] for element in data_all11['train']]
    test_data = [data_all11[element[0]][:] for element in data_all11['test']]
    train11, test11 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all12[element[0]][:] for element in data_all12['train']]
    test_data = [data_all12[element[0]][:] for element in data_all12['test']]
    train12, test12 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view5(OTS)
    train_data = [data_all13[element[0]][:] for element in data_all13['train']]
    test_data = [data_all13[element[0]][:] for element in data_all13['test']]
    train13, test13 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all14[element[0]][:] for element in data_all14['train']]
    test_data = [data_all14[element[0]][:] for element in data_all14['test']]
    train14, test14 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all15[element[0]][:] for element in data_all15['train']]
    test_data = [data_all15[element[0]][:] for element in data_all15['test']]
    train15, test15 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view6(OTS)
    train_data = [data_all16[element[0]][:] for element in data_all16['train']]
    test_data = [data_all16[element[0]][:] for element in data_all16['test']]
    train16, test16 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all17[element[0]][:] for element in data_all17['train']]
    test_data = [data_all17[element[0]][:] for element in data_all17['test']]
    train17, test17 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all18[element[0]][:] for element in data_all18['train']]
    test_data = [data_all18[element[0]][:] for element in data_all18['test']]
    train18, test18 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view7(OTS)
    train_data = [data_all19[element[0]][:] for element in data_all19['train']]
    test_data = [data_all19[element[0]][:] for element in data_all19['test']]
    train19, test19 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all20[element[0]][:] for element in data_all20['train']]
    test_data = [data_all20[element[0]][:] for element in data_all20['test']]
    train20, test20 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all21[element[0]][:] for element in data_all21['train']]
    test_data = [data_all21[element[0]][:] for element in data_all21['test']]
    train21, test21 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view8(OTS)
    train_data = [data_all22[element[0]][:] for element in data_all22['train']]
    test_data = [data_all22[element[0]][:] for element in data_all22['test']]
    train22, test22 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all23[element[0]][:] for element in data_all23['train']]
    test_data = [data_all23[element[0]][:] for element in data_all23['test']]
    train23, test23 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all24[element[0]][:] for element in data_all24['train']]
    test_data = [data_all24[element[0]][:] for element in data_all24['test']]
    train24, test24 = data_g(train_data, test_data, img_width1, img_height1, channels)

    # view9(OTS)
    train_data = [data_all25[element[0]][:] for element in data_all25['train']]
    test_data = [data_all25[element[0]][:] for element in data_all25['test']]
    train25, test25 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all26[element[0]][:] for element in data_all26['train']]
    test_data = [data_all26[element[0]][:] for element in data_all26['test']]
    train26, test26 = data_g(train_data, test_data, img_width1, img_height1, channels)

    train_data = [data_all27[element[0]][:] for element in data_all27['train']]
    test_data = [data_all27[element[0]][:] for element in data_all27['test']]
    train27, test27 = data_g(train_data, test_data, img_width1, img_height1, channels)

    return train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12, train13, \
            train14, train15, train16, train17, train18, train19, train20, train21, train22, train23, train24, train25, \
            train26, train27, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, \
            test13, test14, test15, test16, test17, test18, test19, test20, test21, test22, test23, test24, test25, test26,\
            test27,train_labels,test_labels