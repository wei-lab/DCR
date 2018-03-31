#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)


    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    # origin = np.array(list(reversed(itkimage.GetOrigin())))
    #
    # # Read the spacing along each dimension
    # spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan

# img=load_itk('/home/mi/PycharmProjects/Brain-tumor-segmention/res/VSD.Brain.XX.O.MR_Flair.41186.mha0.mha')
# for line in img:
#     print line


def get_file_dir(data_dir,save_list=None):
    trainfile=open("Testing.txt",'w')
    for dir in os.listdir(data_dir):
      data_obj = [' ' for n in range(4)]
     # mask_obj = [' ']
       # print data_dir+'/'+dir
      for subdir in os.listdir(data_dir+'/'+dir):
        for (dirpath,dirname,filenames) in os.walk(data_dir+'/'+dir+'/'+subdir):

             file= [data_dir+'/'+dir+'/'+subdir+'/'+x for x in filenames if '.mha'in x]
             if len(file)>=1:
                 if 'Flair' in file[0]:
                     data_obj[0]=file[0]

                 elif '_T1c' in file[0]:
                     data_obj[1] = file[0]

                 elif '_T1' in file[0]:
                     data_obj[2] = file[0]

                 elif '_T2' in file[0]:
                     data_obj[3]=file[0]

                 # elif 'more.' in file[0]:
                 #     mask_obj[0]=file[0]
      #data_obj.append(mask_obj[0])
       # print str(data_obj)+('\n')
      trainfile.write(str(data_obj)[1:-1]+'\n')
    trainfile.close()

#
#get_file_dir('../data/Testing/HGG_LGG')


def decode_MRI_png():
    list_file='train.txt'
    with open(list_file,'r') as f:

        for line in f.readlines():
           for em in line.split(','):
               i=0
               em=em.strip()
               for img in load_itk(em.strip()[1:-1]):
                     print (np.shape(load_itk(em.strip()[1:-1])))
                     temp=''.join(e+'/' for e in str(em.replace('HGG','IMGHGG')[1:-1]).split('/')[0:-1])

                     if not os.path.isdir(temp):
                        os.makedirs(temp)
                    # print np.sum(img)
                     img=np.reshape(np.array(img),(240,240))
                     #img=img/10.0
                     if i==64:
                         # print img[100]
                         print (em.replace('HGG','IMGHGG')[1:-1]+str(i)+'.png')
                     # img = Image.fromarray(img)
                     # img= img.convert("L")
                     # img.save(em.replace('HGG','sco')[1:-1]+str(i)+'.png')
                     if 'more.' in em:
                         # npad = ((60 / 2, 60 / 2), (60 / 2, 60 / 2))
                         # img = np.pad(np.array(img), pad_width=npad, mode='constant', constant_values=255)
                         plt.imsave(em.replace('HGG', 'IMGHGG')[1:-1] + str(i) + '.png', img, cmap='gray')
                     else:

                         plt.imsave(em.replace('HGG','IMGHGG')[1:-1]+str(i)+'.png',img, cmap='gray')
                     i=i+1



#decode_MRI_png()

def get_img_dir():
    trainfile=open("Testing.txt",'r')

    trainexample=open('testimgs.txt','w')

    for line in trainfile.readlines():
        line=line.split(',')

        Flair_path=''.join(e +'/' for e in line[0].strip().split('/')[0:-1])
        Flair_path=Flair_path.replace('HGG_LGG','IMGHGG_LGG').replace("'",'')
        Flair_path_filename=line[0].strip().split('/')[-1][:-1]
        T1c_path=''.join(e +'/' for e in line[1].strip().split('/')[0:-1])
        T1c_path=T1c_path.replace('HGG_LGG','IMGHGG_LGG').replace("'",'')
        T1c_path_filename=line[1].strip().split('/')[-1][:-1]
        T1_path=''.join(e +'/' for e in line[2].strip().split('/')[0:-1])
        T1_path=T1_path.replace('HGG_LGG','IMGHGG_LGG').replace("'",'')
        T1_path_filename=line[2].strip().split('/')[-1][:-1]
        T2_path=''.join(e +'/' for e in line[3].strip().split('/')[0:-1])
        T2_path=T2_path.replace('HGG_LGG','IMGHGG_LGG').replace("'",'')
        T2_path_filename=line[3].strip().split('/')[-1][:-1]
        # mask=''.join(e +'/' for e in line[4].strip().split('/')[0:-1])
        # mask_path=mask.replace('HGG_LGG','IMGHGG_LGG').replace("'",'')
        # mask_path_filename=line[4].strip().split('/')[-1][:-1]
        #print Flair_path

        #print Flair_path.replace("'",'')

        for (dirpath, dirname, filenames) in os.walk(Flair_path.replace("'",'')):

            ##print np.sort(filenames)
            for img in filenames:
                tempstr = ''
                flag=img.split('.')[-2]
                #print Flair_path
                #print Flair_path+img+' '
                tempstr+=Flair_path+img+','

                tempstr+=T1c_path+T1c_path_filename[:-4]+'.'+flag+'.png'+','
                tempstr+=(T1_path + T1_path_filename[:-4] + '.' + flag + '.png' + ',')
                tempstr+=(T2_path + T2_path_filename[:-4] + '.' + flag + '.png' + ',')
                #tempstr+=(mask_path + mask_path_filename[:-4] + '.' + flag + '.png' )
                trainexample.write(tempstr+'\n')

    trainexample.close()

    trainfile.close()



#get_img_dir()

def write_img_to_mha(images,mha_name):

    img=sitk.GetImageFromArray(images)
    img.SetOrigin([0, 0, 0])
    img.SetSpacing([1.0, 1.0, 1.0])

    sitk.WriteImage(img,mha_name)

def images_to_mha():
    test_mha=open("Testing.txt",'r')
    for line in test_mha.readlines():
        temp=str(line.split(',')[0]).split('/')
        imgs=[]
        for i in range(155):
            img=Image.open('../res/'+temp[-1][:-1]+str(i)+".png")
            npad = ((48 / 2, 48 / 2), (48 / 2, 48 / 2))
            img = np.pad(np.array(img), pad_width=npad, mode='constant', constant_values=0)
            imgs.append(img//63)
            # print np.array(imgs).dtype

        mha_name='VSD.Seg_HG_001.'+str(temp[-1][:-1]).split('.')[-2]+'.mha'
        write_img_to_mha( np.array(imgs),"../mhares/"+mha_name)

        #print temp[-1][:-1]

def mha_to_mhas():
    test_mha = open("Testing.txt", 'r')
    for line in test_mha.readlines():
        temp = str(line.split(',')[0]).split('/')
        imgs = []
        for i in range(155):
            img=load_itk('../res/'+temp[-1][:-1]+str(i)+".mha")
            # npad = ((48 / 2, 48 / 2), (48 / 2, 48 / 2))
            # img = np.pad(np.array(img), pad_width=npad, mode='constant', constant_values=0)
            imgs.append(img )
            # print np.array(imgs).dtype

        mha_name = 'VSD.Seg_HG_001.' + str(temp[-1][:-1]).split('.')[-2] + '.mha'
        write_img_to_mha(np.array(imgs), "../mhares/" + mha_name)

#mha_to_mhas()





# if __name__=='__main__':
#    #images_to_mha()
#     img = Image.open('/home/mi/PycharmProjects/Brain-tumor-segmention/data/BRATS2015_Training/IMGHGG/brats_tcia_pat124_0003/VSD.Brain_3more.XX.O.OT.42299/VSD.Brain_3more.XX.O.OT.42299.mha50.png')
#     img=img.convert('L')
#     for i in range(70,170):
#         print np.array(img)[i]
        #print np.array(img)[110]
    #print np.array(load_itk('/home/mi/PycharmProjects/Brain-tumor-segmention/mhares/VSD.Seg_HG_001.40465.mha')[100][90]).dtype




