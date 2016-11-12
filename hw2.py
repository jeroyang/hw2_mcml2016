import os
import random
import time
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


np.set_printoptions(threshold=np.nan)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#def pca_dim_reduction(faces_1d_list, n):
#    pca = PCA(n_components = n)
#    pca.fit(faces_1d_list)
#    return pca.transform(faces_1d_list)


if __name__ == '__main__':
    image_list_g1d = []
    image_label = []
    
    t_start = time.time()   # timer
    new_w = 3   # new width

    print(
'''
---------Prepocessing-------------
Resizeing the images into''',new_w,"x",new_w,'''
----------------------------------
''')

    for filename in os.listdir('CSL/training/'):
        if isfile(join('CSL/training/', filename)) and filename != ".DS_Store":
            im = Image.open("CSL/training/{}".format(filename))            
            im_re = plt_img.pil_to_array(im.resize((new_w, new_w), Image.ANTIALIAS))    # resize the image, and save as <uint8>
            image_list_g1d.append(np.reshape(rgb2gray(im_re),-1))
            image_label.append(filename[0:1])
    
    p_tmp = time.time() - t_start   # timer
    t_tmp = time.time()   # timer
    print("(Read data finished, time used:", "{0:.2f}".format(p_tmp), "seconds.)")   # timer

    
    print(
'''
---------Classification-----------
method: Linear SVM
----------------------------------
''')

    n_fold = 10
    data_size = len(image_list_g1d)
    test_size = int(data_size/10)
    rnd_list = list(range(0, data_size))
    random.shuffle(rnd_list)
    acc_list = []
    

    for j in range(0, n_fold):        
        test_set = [image_list_g1d[i] for i in rnd_list[test_size*j:test_size*(j+1)]]
        test_label = [image_label[i] for i in rnd_list[test_size*j:test_size*(j+1)]]
        
        X = [image_list_g1d[i] for i in (rnd_list[:test_size*j] +rnd_list[test_size*(j+1):])]
        y = [image_label[i] for i in (rnd_list[:test_size*j] +rnd_list[test_size*(j+1):])]
        
        '''SVM'''
        clf = svm.LinearSVC()
        clf.fit(X,y)
        predicted = clf.predict(test_set)
        acc_list.append(accuracy_score(test_label, predicted))
        print("#",j+1, ") Accuracy:", acc_list[j])
        print(classification_report(test_label, predicted))


    ''' final results'''
    acc_avg = sum(acc_list)/ float(n_fold)
    acc_list_fomatted = [float("{0:.2f}".format(x)) for x in acc_list]
    print(
'''
--------------------------Final result------------------------------
> Prepocessing:   Resize images into''',new_w,"x",new_w,'''
> Classification: Linear SVC
> Evaluation:     10-fold Cross Validation

> Accuray:''', acc_list_fomatted,'''

> Average Accuracy:''', "{0:.2f}".format(acc_avg),'''
---------------------------------------------------------------------
''')

    p_total = time.time() - t_start   # timer
    print("(Total time used:", "{0:.2f}".format(p_total), "seconds.)")       # timer


#    '''testing data'''
#    test_set = []
#    test_label = []
#    
#    for filename in os.listdir('CSL/test/'):
#        if isfile(join('CSL/test/', filename)) and filename != ".DS_Store":
#            #im=plt.imread("CSL/test/{}".format(filename))
#            im = Image.open("CSL/test/{}".format(filename))            
#            im_re = plt_img.pil_to_array(im.resize((64,64), Image.ANTIALIAS))
#            test_set.append(np.reshape(rgb2gray(im_re),-1))
#            test_label.append(filename[0:1])

#    
#    '''
#    plt.subplot(211)
#    plt.imshow(image_list[0])
#    plt.subplot(212)
#    plt.imshow(image_list_g[1],'gray')
#    plt.show()
#    '''
