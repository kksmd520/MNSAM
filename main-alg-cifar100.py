'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
from torchvision import datasets
import os
import torch.utils.data as data
import datetime
import datetime
import argparse
from sam import SAM,DSAM,enable_running_stats,disable_running_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adan import Adan
from numpy import *
from models import *
from models.vgg import vgg11_bn
from sum import SUM
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.pyplot import MultipleLocator
from torchvision import models

# 定义随机数, 复现结果    
def set_seed(seed):  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fun(worker_id, seed):
    np.random.seed(seed + worker_id)



def train(net, optimizer, train_loader, device, epoch, criterion,opt,scheduler):

    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loader_num=len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs_dev, targets_dev = inputs.to(device), targets.to(device)
        if optimizer.__class__.__name__ == 'SAM' or optimizer.__class__.__name__ == 'DSAM':
            enable_running_stats(net)
            outputs = net(inputs_dev)  # 通过前向传播获取网络输出
            loss = criterion(outputs, targets_dev)  # 计算损失
            loss.backward()  # 自动计算梯度
            optimizer.first_step(zero_grad=True)
            # second forward-backward step
            disable_running_stats(net)
            criterion(net(inputs_dev), targets_dev).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            outputs = net(inputs_dev)
            loss = criterion(outputs, targets_dev)
            loss.backward()
            optimizer.step()

        train_loss+=loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_dev).sum().item()


    train_loss /= loader_num
    tain_acc=100.*correct / len(train_loader.dataset)

    print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(inputs), len(train_loader.dataset),tain_acc, train_loss))

    return train_loss,tain_acc
      


def test(net, test_loader, device, epoch, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loader_num=len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs_dev, targets_dev = inputs.to(device), targets.to(device)
            outputs = net(inputs_dev)
            loss = criterion(outputs, targets_dev)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_dev.size(0)
            correct += predicted.eq(targets_dev).sum().item()
            
            
    test_loss /= loader_num 
    test_acc=100.*correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),test_acc))
    
    return test_loss,test_acc 



def create_data(worker_id, fname, seed):

    if fname == 'cifar-10':
        print('==> Preparing data cifar10..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
   
    elif fname == 'cifar-100':
        print('==> Preparing data cifar100..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507,0.487,0.441), (0.267,0.256,0.276)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507,0.487,0.441), (0.267,0.256,0.276)),
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2,worker_init_fn=worker_init_fun(worker_id, seed))
    elif fname == 'tiny-imagenet':
        print('==> Preparing data tiny-imagenet..')
        data_dir = './data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(data_dir + '/train', transform=transforms.ToTensor())
        train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
        print(len(train_dataset))
        test_dataset = datasets.ImageFolder(data_dir + '/val', transform=transforms.ToTensor())
        test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=True)
    
    else:        
          raise ValueError("Invalid cifar")
          
    return train_loader, test_loader




def create_model(fname, device):
    if fname == 'cifar-10':
        num_classes = 10
        net = ResNet34(num_classes)

    elif fname  == 'cifar-100':
        num_classes = 100
        net = ResNet34(num_classes=num_classes)
        # net = vgg11_bn(3, num_classes)
    elif fname == 'tiny-imagenet':
        # num_classes = 200
        # net = vit.__dict__['vit_small_patch16'](
        #     img_size=64,
        #     num_classes=num_classes,
        #     drop_path_rate=0.1,
        #     global_pool=True,
        # )
        net =models.resnet50()
    else:
        raise ValueError("Invalid model")
    
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    print('==> Building model..')
      
    return net




def create_optimizer(opt, net, weight_decay, T):

    if opt == 'Adagrad':
        optimizer= torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif opt == 'RMSprop':
        optimizer= torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif opt == 'SGD':
        optimizer= torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif opt == 'SUM-0.0':
        optimizer= SUM(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.1, interp_factor=0.0, K=K*T)
    
    elif opt == 'SUM-0.5':
        optimizer= SUM(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.1, interp_factor=0.5, K=K*T)
    
    elif opt == 'SUM-1.0':
        optimizer= SUM(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.1, interp_factor=1.0, K=K*T)
    elif opt =='dsam-0.001':
        optimizer = DSAM(net.parameters(), torch.optim.SGD, rho=0.001, adaptive=False, lr=lr,
                         momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt =='dsam-0.005':
        optimizer = DSAM(net.parameters(), torch.optim.SGD, rho=0.005, adaptive=False, lr=lr,
                         momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt == 'dsam-0.01':
        optimizer = DSAM(net.parameters(), torch.optim.SGD, rho=0.01, adaptive=False, lr=lr,
                         momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt =='dsam-0.05':
        optimizer = DSAM(net.parameters(), torch.optim.SGD, rho=0.05, adaptive=False, lr=lr,
                         momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt == 'adam':
        optimizer= torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay,betas=(0.9,0.99))
    elif opt == 'sam':
        optimizer = SAM(net.parameters(), torch.optim.SGD, rho=0.001, adaptive=False, lr=lr,
                        momentum=0, weight_decay=weight_decay, nesterov=False)
    elif opt == 'asam':
        optimizer = SAM(net.parameters(), torch.optim.SGD, rho=0.001, adaptive=True, lr=lr,
                        momentum=0, weight_decay=weight_decay, nesterov=False)
    elif opt =='adan':
        optimizer = Adan(net.parameters(), lr=lr)
    else:
        print("请输入正确的训练算法")
    
    return optimizer

    

def train_and_save(trial_idx,start_epoch):
    for opt in algorithms:                
        train_loader, test_loader = create_data(worker_id, fname, seed)
        T = len(train_loader)*(epochs)
        net = create_model(fname, device)
        criterion = nn.CrossEntropyLoss() 
        
        optimizer = create_optimizer(opt, net, weight_decay, T)
        
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, last_epoch=-1)   
        paras = ('dsets=%s trial_idx=%s opt=%s epochs=%s batch_size=%s lr=%s K=%s seed=%s' %
                (fname, trial_idx, opt, epochs, batch_size, lr, K, seed))
        history_file = os.path.join(history_folder,
                                    paras+'.csv')
        ckpt_path = './checkpoint/' + paras + '.pth'
        if os.path.exists(history_file):
            print("tests目录下已经存在相同csv文件，请备份以免覆盖,如果已经备份请删除该文件！")
            break
        
        if resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(ckpt_path)
            net.load_state_dict(checkpoint['net'])
            # test_acc = checkpoint['acc']
            info = checkpoint['info']
            start_epoch = checkpoint['epoch']
            print("当前checkpoint信息为：", info)
            
        M_columns = ['train_loss_'+opt, 'train_acc_'+opt, 'test_loss_'+opt, 'test_acc_'+opt]
        columns_total = ['epoch'] + M_columns
        dfhistory = pd.DataFrame(columns=columns_total)
           
        start = datetime.datetime.now()
        for epoch in range(start_epoch, start_epoch+epochs):
            M_infos = tuple()

            if not debug:
                tr_loss, tr_metric=train(net, optimizer, train_loader, device, epoch, criterion, opt, scheduler)
                te_loss, te_metric = test(net, test_loader, device, epoch, criterion)
            else:
                tr_loss, tr_metric=0.1,99
                te_loss, te_metric=1.0,66
                
            M_infos += (tr_loss, tr_metric, te_loss, te_metric)    
            info = (int(epoch),) + M_infos
            dfhistory.loc[int(epoch)] = info
            # print("[epoch = %d] loss: %.5f, acc: %.3f" % (info[:3]))     
        
            # Save checkpoint.
            state = {
                'net': net.state_dict(),
                'acc': te_metric,
                'epoch': epoch,
                'info': paras
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, ckpt_path) 
                
            dfhistory.to_csv(history_file)
            torch.cuda.empty_cache()
            
            end = datetime.datetime.now()
            print('time%.3f:'%(((end-start).seconds)/60))
    
if __name__ == '__main__':
    
##########################调整参数#######################
    start_idx = 3
    trial_num = 4
    seed = 1
    worker_id = 0
    epochs = 10
    lr = 0.01
    T_max =50
    beta1 = 0.9
    beta2 = 0.999
    interp_factor = 1.0
    K = 0.9
    batch_size = 128
    start_epoch = 0
    weight_decay = 0 
    resume = False
    fname = "ci'fa'r"
    debug= False
    # algorithms = ['Adagrad', 'RMSprop', 'SGD', 'SUM-0.0', 'SUM-0.5', 'SUM-1.0']
    algorithms = ['adan','adam','dsam-0.01','SGD','asam','sam']
    # algorithms = ['dsam-0.01']
##########################调整参数#######################
    

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ROOT_PATH = os.path.abspath('./')
    history_folder = os.path.join(ROOT_PATH, 'results/alg-tiny-imagenet')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for trial_idx in range(start_idx,trial_num+start_idx):
        seed = trial_idx*5
        set_seed(seed)
        train_and_save(trial_idx,start_epoch)    
