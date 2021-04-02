# import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy.io as sio
import h5py
import numpy as np
from keras.utils import np_utils
import scipy
import random

# Get ImageDataGenerator arguments(options) depends on train
def datagen_gen():
    datagen = ImageDataGenerator( rotation_range=180,
                            horizontal_flip=True,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            vertical_flip=True,
                            fill_mode='nearest')

    return datagen

nam = 'data/LIDC_nodule_d3_all_9'
f = 'original'
# f = 'texture'
# f = 'shape'
for k in range(1,11):
    for v in range(1,10):
        data_all = h5py.File(nam + '/nodule_' + str(k) + '_' + str(v) + '_' + f + '.mat')

        train_data = [data_all[element[0]][:] for element in data_all['train']]
        test_data = [data_all[element[0]][:] for element in data_all['test']]
        train_labels = data_all['labels_tr'][:].T
        test_labels = data_all['labels_te'][:].T

        #train
        num = len(train_data)
        train = []
        labels = []
        for ii in range(num):
            xx = np.empty((1,train_data[ii].shape[0],train_data[ii].shape[1],1), dtype="double")
            xx[0,:,:,0] = train_data[ii]
            
            datagen = datagen_gen()

            i = 0
            train.append(scipy.misc.imresize(train_data[ii], (224, 224)))
            labels.append(train_labels[ii])
            # scipy.misc.imsave('data_argu/'+str(k)+'_'+str(v)+'/original'+str(ii)+'.jpg',  xx[0,:,:,0]) #save original patch
            for batch in datagen.flow(xx,batch_size=1):
                                      # save_to_dir='data_argu/'+str(k)+'_'+str(v),#save augmented patch
                                      # save_prefix='original_argu'+str(ii),
                                      # save_format='jpg'):
                train.append(scipy.misc.imresize(batch[0,:,:,0], (224, 224)))
                labels.append(train_labels[ii])
                i += 1
                if i > 3: #four times
                    break  # otherwise the generator would loop indefinitely

        labels =  np.array(labels)
        sio.savemat(('data/LIDC_nodule_d3_all_9_agu_224/nodule_' + str(k) + '_' + str(v) + '_' + f + '.mat'),
                    {'train': train, 'labels_tr': labels, 'test': test_data, 'labels_te': test_labels})
