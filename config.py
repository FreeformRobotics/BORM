import os
import shutil
import numpy as np

path='/data/cenj/SUNRGBD/xtion/xtion_align_data'

file_name='categories_places365_home.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
print(classes)

def move_file(dir):
    if os.path.isdir(dir):
        for s in os.listdir(dir):
#            s_1=os.path.join(dir,s)
#            for ss in os.listdir(s_1):
#                s_2=os.path.join(s_1,ss)
#                for sss in os.listdir(s_2):
                    u = os.path.join(dir, s)
                
                    img_dir=os.path.join(u,'image')
                    print(u)
                    for t in os.listdir(img_dir):
                        if 'jpg' in t:
                            img=os.path.join(img_dir,t)
                    print(img)
                    for t in os.listdir(u):
                        if t == 'scene.txt':
                            file=open(os.path.join(u, t),'r')
                            file_data=file.readlines()
                            for catogory in classes:
                                if file_data[0] == catogory:
                                    shutil.copy(img, '/data/cenj/SUNRGBD_val/'+catogory)

                    
                    
                    
if __name__ == '__main__':
    move_file(path)
