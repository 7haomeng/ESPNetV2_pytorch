import cv2
import os

def readFile():
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''
        num = 0
        with open('./dataSet/val_hao.txt', 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split(',')
                img_file = ('./dataSet/' + line_arr[0].strip()).strip()
                label_file = ('./dataSet/' + line_arr[1].strip()).strip()
                rgb_img = cv2.imread(img_file)
                label_img = cv2.imread(label_file)
                # label_img = cv2.imread(label_file, 0)
                rgb_img = cv2.resize(rgb_img, (640, 320), interpolation=cv2.INTER_NEAREST)
                label_img = cv2.resize(label_img, (640, 320), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite('./dataSet/hao/val/rgb/'+'rgb_'+str(num)+'.jpg',rgb_img)
                cv2.imwrite('./dataSet/hao/val/label/'+'label_'+str(num)+'.png',label_img)
                num = num + 1

        return 0

if __name__ == '__main__':
    readFile()