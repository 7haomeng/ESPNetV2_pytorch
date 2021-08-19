#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import Image

import numpy as np
import torch
import glob

import sys
import os
from argparse import ArgumentParser
from cnn import SegmentationModel as net
from torch import nn

import timeit

# from PIL import Image
from collections import OrderedDict

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append('/home/chinghaomeng/cv_bridge_python3/install/lib/python3/dist-packages')
import cv2
from cv_bridge import CvBridge, CvBridgeError

#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

class Prediction:
    def __init__(self, args):
        self.cv_bridge = CvBridge()

        self.modelA = net.EESPNet_Seg(args.classes, s=args.s)

        if not os.path.isfile(args.pretrained):
            print('Pre-trained model file does not exist. Please check ./pretrained_models folder')
            exit(-1)
        self.modelA = nn.DataParallel(self.modelA)
        self.modelA.load_state_dict(torch.load(args.pretrained))


        if torch.cuda.is_available():
            print("GPU : {0}\n".format(torch.cuda.is_available()))
            self.modelA = self.modelA.cuda()

        # set to evaluation mode
        self.modelA.eval()

        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)

        self.pallete = [[0, 0, 0],
                [99, 197, 34],
                [153, 153, 153],
                [250, 170, 30],
                [250, 170, 30],
                [70, 130, 180],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [128, 64, 128],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [244, 35, 232],
                [220, 20, 60],
                [0, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [0, 0, 0]]

        self.img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.predict_cb)
        self.predict_pub = rospy.Publisher("/ESPNet_v2/predict_img", Image, queue_size=1)

    def predict_cb(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")    # Convert ros image topic to cv_image
        predict = self.evaluateModel(args, self.modelA, cv_image)
        self.predict_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict, "bgr8"))


    def relabel(self, img):
        '''
        This function relabels the predicted labels so that cityscape dataset can process
        :param img:
        :return:
        '''
        img[img == 19] = 255
        img[img == 18] = 33
        img[img == 17] = 32
        img[img == 16] = 31
        img[img == 15] = 28
        img[img == 14] = 27
        img[img == 13] = 26
        img[img == 12] = 25
        img[img == 11] = 24
        img[img == 10] = 23
        img[img == 9] = 22
        img[img == 8] = 21
        img[img == 7] = 20
        img[img == 6] = 19
        img[img == 5] = 17
        img[img == 4] = 13
        img[img == 3] = 12
        img[img == 2] = 11
        img[img == 1] = 8
        img[img == 0] = 100#7
        img[img == 255] = 0
        return img


    def evaluateModel(self, args, model, image):
        # gloabl mean and std values

        mean = [107.82763, 108.5122, 112.27358]
        std = [55.42822, 54.48955, 51.91889]

        model.eval()

        start = timeit.default_timer()
        img = image
        # img = cv2.imread(image)             # No need

        if args.overlay:
            img_orig = np.copy(img)

        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        # resize the image to 1024x512x3
        img = cv2.resize(img, (args.inWidth, args.inHeight))
        if args.overlay:
            img_orig = cv2.resize(img_orig, (args.inWidth, args.inHeight))
        
        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        if args.gpu:
            img_tensor = img_tensor.cuda()
        img_out = model(img_tensor)

        stop = timeit.default_timer()
        # print("The num of rgb_img: {0}".format(i))
        print("Time: {0}\n".format(stop-start))
        classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
        # if i % 100 == 0 and i > 0:
            # print('Processed [{}/{}]'.format(i, len(image_list)))
        # name = image.split('/')[-1]
        
        if args.colored:
            classMap_numpy_color = np.zeros((img.shape[1], img.shape[2], img.shape[0]), dtype=np.uint8)
            for idx in range(len(self.pallete)):
                [r, g, b] = self.pallete[idx]
                classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
            # cv2.imwrite(args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png'), classMap_numpy_color)
            if args.overlay:
                overlayed = cv2.addWeighted(img_orig, 1.0, classMap_numpy_color, 0.5, 0)
                # cv2.imwrite(args.savedir + os.sep + 'over_' + name.replace(args.img_extn, 'jpg'), overlayed)
                return overlayed
        if args.cityFormat:
            classMap_numpy = self.relabel(classMap_numpy.astype(np.uint8))
        # cv2.imwrite(args.savedir + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)

    def onShutdown(self):
        rospy.loginfo("Shutdown.")


if __name__ == '__main__':
    rospy.init_node('semantic', anonymous=True)

    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--data_dir', default="./dataSet/hao/test/rgb", help='Data directory')
    parser.add_argument('--img_extn', default="jpg", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=640, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=480, help='Height of RGB image')#320
    parser.add_argument('--savedir', default='./dataSet/hao/results_hao/results_test', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--pretrained', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=0.5, type=float, help='scale')
    parser.add_argument('--cityFormat', default=False, type=bool, help='If you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    
    prediction = Prediction(args)
    rospy.on_shutdown(prediction.onShutdown)

    rospy.spin()
