import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from OLE import *
import densenet as dn
from operator import itemgetter
from AngularLoss import *
from temperature_scaling import *

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))
    
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #define loss function (criterion) and pptimizer
    criterion = [nn.CrossEntropyLoss().cuda()] + [OLELoss(lambda_=.5)] + [AngleLoss()]#+ [OLELoss(lambda_=.25)]
    '''
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    '''
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def binning(x):
    sorted(x,key=itemgetter(0))
    #print(x)
    index = np.zeros((15,3))
    x[:,0]*=15
    sum=0
    for i in range(x.shape[0]):
        if(math.floor(x[i,0]>=15)):
            index[14,0]+=1
        else:
            index[math.floor(x[i,0]),0]+=1
    for i in range(index.shape[0]):
        for j in range(int(index[i,0])):
            if(x[int(sum)+j,1]==x[int(sum)+j,2]):
                index[i,1] +=1
        sum +=index[i,0]
    for i in range(index.shape[0]):
        if(index[i,0]!=0):
            index[i,2]=index[i,1]/index[i,0] 
    print("Before Scaling")
    print(index)

def log_back(x):
    x= np.asarray(x)
    x = np.exp(x)
    sum_den = np.sum(x,axis=1)
    #print(sum_den)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = (x[i,j]/sum_den[i])
    return x

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        losses_list=torch.cuda.FloatTensor([-1,1,1])
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        loss=0
        # compute output
        features,output,param = model(x = input_var,target = target_var)
        cos_theta,phi_theta = features
        #print(cos_theta.shape)
        for cix, crit in enumerate(criterion):
            #NLL
            if(cix==0):
                losses_list[cix] = crit(output.cuda(), target_var)
                loss += losses_list[cix]
                  #print("cix==0")
            
            #OLE Loss
            elif(cix==1):
                losses_list[cix] = crit(cos_theta.cuda(), target_var)
                loss += .5*losses_list[cix]
            #Angular Loss
            else:
                #print(output)
                losses_list[cix] = crit(features, target_var)
                #print(losses_list[cix])
                #loss += losses_list[cix]

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.9f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
            #binning(features)
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    x =np.zeros((10000,3))
    j=0
    b_size = 64
    for i, (input, target) in enumerate(val_loader):
        loss=0
        losses_list=torch.cuda.FloatTensor([-1,1,1])
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        #input_var = torch.Tensor(input_var)
        # compute output
        features,output,param = model(input_var)
        #print(output)
        #print(output.type)
        #output = torch.FloatTensor(output)
        for cix, crit in enumerate(criterion):   
            if(cix==0):
                losses_list[cix] = crit(output.cuda(), target_var)
                loss += losses_list[cix]
                  #print("cix==0")
            elif(cix==1):
                losses_list[cix] = crit(param.cuda(), target_var)
                loss += .5*losses_list[cix]
            else:
                #print(output)
                losses_list[cix] = crit(features, target_var)
                #print(losses_list[cix])
                #loss += losses_list[cix]


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        if(10000-j>64):
            #print(log_back(output.data).max(1)[0])
            x[j:j+b_size,0]= np.asarray(log_back(output.data).max(1)[0])
            x[j:j+b_size,1]= target
            x[j:j+b_size,2] = output.data.max(1)[1]
            #print(log_back(output.data).max(1)[0])
        else:
           #print("Last")
            x[j:10000,0]= np.asarray(log_back(output.data).max(1)[0])
            x[j:10000,1]= target
            x[j:10000,2] = output.data.max(1)[1]
        #print(output.data.max(1)[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        j += b_size

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        #print(output.data.max(1)[0])
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(val_loader)
    binning(x)
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 10 and 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10)) * (0.1 ** (epoch // 15))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()