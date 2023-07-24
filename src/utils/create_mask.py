import numpy as np
import os
import json
import sys
import pprint
import random
import shutil
from PIL import Image
import glob
import copy
import torch
import cv2

# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

Dataset_maskPath='/share/users/sanoojan/e4s/data/CelebAMask-HQ/CelebA-HQ-mask'
save_path='/share/users/sanoojan/e4s/data/CelebAMask-HQ/CelebA-HQ-mask/Overall_mask'
for i in range(30000):
    #  create blank image with 512,512

    mask=np.zeros((512,512))
    for ind,cate in enumerate(celelbAHQ_label_list):
        # check if path exists s.path.join(Dataset_maskPath,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')
        if os.path.exists(os.path.join(Dataset_maskPath,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')):
            im= Image.open(os.path.join(Dataset_maskPath,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')).convert('L')
            im = np.equal(im, 255)
            mask[im]=ind+1
     
    # save the mask
    cv2.imwrite(os.path.join(save_path,'%d'%i+'.png'),mask)


