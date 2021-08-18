import datetime
# from torchvision import transforms
import loadData_hao as ld
import os
import torch
import pickle
from cnn import SegmentationModel as net
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet_hao as myDataLoader
from argparse import ArgumentParser
from train_utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

def trainValidateSegmentation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.EESPNet_Seg(args.classes, s=args.s, pretrained=args.pretrained, gpus=num_gpus)

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)

    date = datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second

    args.savedir = args.savedir + str(args.s) + "_" + str(date) + "_" + str(hour) + "-" + str(minute) + "-" + str(args.s)+ '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))



    if cuda_available:
        args.onGPU = True
        ## my set
        torch.cuda.set_device(0)   
        model = model.cuda()

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if args.onGPU:
        weight = weight.cuda()

    criteria = torch.nn.CrossEntropyLoss(weight) #weight

    if args.onGPU:
        criteria = criteria.cuda()

    files = os.listdir('./dataSet/hao/weight')
    listText_w = open('./dataSet/hao/weight'+'/'+'weight_hao.txt', 'w')
    data_statistics = 'Data statistics :'+'\n'+'mean : '+str(data['mean'])+','+'std : '+str(data['std'])+'\n'+'classWeights : '+str(data['classWeights'])+'\n'
    listText_w.write(data_statistics)
    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])

    #compose the data with transforms

    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(args.inWidth, args.inHeight)),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64).
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # trainDataset_main = transforms.Compose([
    #     transforms.Normalize(mean=data['mean'], std=data['std']),
    #     transforms.RandomResizedCrop(size=(args.inWidth, args.inHeight)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ColorJitter(brightness=(0, 1), contrast=(0, 1), saturation=(0, 1), hue=(-0.5, 0.5)),
    #     #myTransforms.RandomCrop(64).
    #     transforms.ToTensor(),
    #     #
    # ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*1.5), int(1.5*args.inHeight))),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # trainDataset_scale1 = transforms.Compose([
    #     transforms.Normalize(mean=data['mean'], std=data['std']),
    #     transforms.RandomResizedCrop(size=(int(args.inWidth*1.5), int(1.5*args.inHeight))),
    #     transforms.RandomHorizontalFlip(p=0.9),
    #     transforms.RandomVerticalFlip(p=0.9),
    #     transforms.ColorJitter(brightness=(0, 8), contrast=(0, 8), saturation=(0, 8), hue=(-0.4, 0.4)),
    #     transforms.ToTensor(),
    #     #
    # ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*1.25), int(1.25*args.inHeight))), # 1536, 768
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # trainDataset_scale2 = transforms.Compose([
    #     transforms.Normalize(mean=data['mean'], std=data['std']),
    #     transforms.RandomResizedCrop(size=(int(args.inWidth*1.25), int(1.25*args.inHeight))), # 1536, 768
    #     transforms.RandomHorizontalFlip(p=0.7),
    #     transforms.RandomVerticalFlip(p=0.7),
    #     transforms.ColorJitter(brightness=(0, 6), contrast=(0, 6), saturation=(0, 6), hue=(-0.3, 0.3)),
    #     transforms.ToTensor(),
    #     #
    # ])

    trainDataset_scale3 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*0.75), int(0.75*args.inHeight))),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # trainDataset_scale3 = transforms.Compose([
    #     transforms.Normalize(mean=data['mean'], std=data['std']),
    #     transforms.RandomResizedCrop(size=(int(args.inWidth*0.75), int(0.75*args.inHeight))),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.ColorJitter(brightness=(0, 4), contrast=(0, 4), saturation=(0, 4), hue=(-0.2, 0.2)),
    #     transforms.ToTensor(),
    #     #
    # ])

    trainDataset_scale4 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*0.5), int(0.5*args.inHeight))),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # trainDataset_scale4 = transforms.Compose([
    #     transforms.Normalize(mean=data['mean'], std=data['std']),
    #     transforms.RandomResizedCrop(size=(int(args.inWidth*0.5), int(0.5*args.inHeight))),
    #     transforms.RandomHorizontalFlip(p=0.3),
    #     transforms.RandomVerticalFlip(p=0.3),
    #     transforms.ColorJitter(brightness=(0, 2), contrast=(0, 2), saturation=(0, 2), hue=(-0.1, 0.1)),
    #     transforms.ToTensor(),
    #     #
    # ])
    

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # valDataset = transforms.Compose([
    #     transforms.Normalize(mean=data['mean'], std=data['std']),
    #     transforms.Scale(args.inWidth, args.inHeight),
    #     transforms.ToTensor(),
    #     #
    # ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale3 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale4 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    start_epoch = 0
    best_val = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    ##scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)



    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()

    for epoch in range(start_epoch, args.max_epochs):

        #scheduler.step(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        ## torch.cuda.empty_cache()  
        train(args, trainLoader_scale1, model, criteria, optimizer, epoch)

        train(args, trainLoader_scale2, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale4, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale3, model, criteria, optimizer, epoch)

        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train(args, trainLoader, model, criteria, optimizer, epoch)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)


        is_best = mIOU_val > best_val
        best_val = max(mIOU_val, best_val)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            'best_val': best_val,
        }, args.savedir + 'checkpoint.pth.tar')

        #save the model also
        if is_best:
            model_file_name = args.savedir + os.sep + 'model_best.pth'
            torch.save(model.state_dict(), model_file_name)

        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--data_dir', default="./dataSet", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=640, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=320, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=1500, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./dataSet/hao/espnetv2/results_espnetv2_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')  #
    parser.add_argument('--classes', type=int, default=2, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--s', default=1, type=float, help='scaling parameter')

    trainValidateSegmentation(parser.parse_args())

