import random
import numpy as np
import os

def sample(dir,num):
    files = os.listdir(dir)
    listText_t = open(dir+'/'+'train_hao.txt', 'w')
    listText_v = open(dir+'/'+'val_hao.txt', 'w')

    items = np.arange(0,num)
    random.shuffle(items)
    

    for i in range(int(round(num*0.7))):
        if(items[i] < 10):
            name = 'hao/images/'+'frame000'+str(items[i])+'.jpg'+', '+'hao/masks/'+'frame000'+str(items[i])+'.png'+'\n'
        elif(items[i] >= 10 and items[i]< 100):
            name = 'hao/images/'+'frame00'+str(items[i])+'.jpg'+', '+'hao/masks/'+'frame00'+str(items[i])+'.png'+'\n'
        elif(items[i] >= 100 and items[i] < 1000):
            name = 'hao/images/'+'frame0'+str(items[i])+'.jpg'+', '+'hao/masks/'+'frame0'+str(items[i])+'.png'+'\n' 
        listText_t.write(name)

    for i in range(int(round(num*0.7)), num-1):
        if(items[i] < 10):
            name = 'hao/images/'+'frame000'+str(items[i])+'.jpg'+', '+'hao/masks/'+'frame000'+str(items[i])+'.png'+'\n'
        elif(items[i] >= 10 and items[i]< 100):
            name = 'hao/images/'+'frame00'+str(items[i])+'.jpg'+', '+'hao/masks/'+'frame00'+str(items[i])+'.png'+'\n'
        elif(items[i] >= 100 and items[i]< 1000):
            name = 'hao/images/'+'frame0'+str(items[i])+'.jpg'+', '+'hao/masks/'+'frame0'+str(items[i])+'.png'+'\n'
        listText_v.write(name)

if __name__ == '__main__':
    sample('/home/syntec/ESPNetv2-master/segmentation/dataSet',501)
