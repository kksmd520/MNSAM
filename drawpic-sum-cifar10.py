import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from scipy.ndimage import uniform_filter1d

fname = 'cifar-10'
trial_idxs = [2]
root_path = os.path.abspath('./')
data_path = root_path + '/results/sum-cifar10/'
files = os.listdir(data_path)
len_files = len(files)


train_loss_dict = {}
train_acc_dict = {}
test_loss_dict ={}
test_acc_dict = {}


algo_nme = []


opt_num = 0
tridxs = [ 'trial_idx='+str(k) for k in trial_idxs]

for i in range(len_files):
    idvalues = files[i].split( )
    IsShow = False
    for j, x in enumerate(idvalues):
        if x in tridxs:
           IsShow=True
           break  
    if not IsShow:
        continue 
    
    optname = ''
    for _, x in enumerate(idvalues):         
        if x.startswith('opt='):
            optname = x[4:]
            break
    
    if optname == '':
        continue
    elif optname not in algo_nme:
        algo_nme.append(optname)
      
    for _, x in enumerate(idvalues):  
        if x.endswith(fname):
            data = pd.read_csv(data_path+files[i],sep = ',').values[:,2:]
            if optname in train_loss_dict.keys():
                train_loss_dict[optname].append(data[:,0])
                train_acc_dict[optname].append(data[:,1])
                test_loss_dict[optname].append(data[:,2])
                test_acc_dict[optname].append(data[:,3])
            else:
                train_loss_dict[optname] = [data[:,0]]
                train_acc_dict[optname] = [data[:,1]]
                test_loss_dict[optname] = [data[:,2]]
                test_acc_dict[optname] = [data[:,3]]
                
            break
                

opt_num = len(algo_nme)


plt.figure(num=1,dpi=300)
custom_cycler=(cycler(color = ['g','m','c','r','b','y','m','g'])+
               cycler(linestyle = ['-', '-', '-', '-', '-', '-', '-.','--'])+
               cycler(marker = ['', '', '', '', '', '', '','']))
  
      
plt.rc('axes', prop_cycle = custom_cycler)




##########  绘制 Training Accuracy 图像  ########## 
plt.figure(num=1,dpi=300)
for alg in algo_nme:
    x = np.arange(0,len(train_acc_dict[alg][0]))+1
    y= np.mean(train_acc_dict[alg],0)
    
    label = alg.replace("dsam-", " $\\eta$=")
    plt.plot(x, y, label=label, linewidth=1.5)

plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
ax=plt.gca();  ### 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);  ### 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ### 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);  ### 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);  ### 设置上部坐标轴的粗细

# plt.ylim(10,100)  ### 调整 y 轴坐标
# plt.xlim(-2,104)  ### 调整 x 轴坐标
plt.legend()
plt.grid()
plt.savefig(data_path+fname+'-train-acc.eps',dpi=300)  ### 保存图片
plt.show()
plt.close()




##########  绘制 Training Loss 图像  ##########
plt.figure(num=1,dpi=300)
for alg in algo_nme:
    x = np.arange(0,len(train_loss_dict[alg][0]))+1
    y = np.mean(train_loss_dict[alg],0)
    
    label = alg.replace("dsam-", " $\\eta$=")
    plt.plot(x, y, label= label, linewidth=1.5) 


plt.xlabel("Epochs")
plt.ylabel("Training Loss")
ax=plt.gca();  ### 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);  ### 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ### 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);  ### 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);  ### 设置上部坐标轴的粗细

# plt.ylim(-0.02,1.4)  ### 调整 y 轴坐标
# plt.xlim(-2,104)  ### 调整 x 轴坐标
plt.legend()
plt.grid()
plt.savefig(data_path+fname+'-train-loss.eps',dpi=300)  ### 保存图片
plt.show()
plt.close()

 

     
##########  绘制 Test Accuracy 图像  ##########
plt.figure(num=4,dpi=300)
for index,alg in enumerate(algo_nme):
    x = np.arange(0,len(test_acc_dict[alg][0]))+1 
    y= np.mean(test_acc_dict[alg],0)
    y_smooth = uniform_filter1d(y, size=5)
    label = alg.replace("dsam-", " $\\eta$=")
    plt.plot(x,y_smooth, label=label, linewidth=1.5)


plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
ax=plt.gca();  ### 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);  ### 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ### 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);  ### 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);  ###  设置上部坐标轴的粗细

# plt.ylim(60,95)  ### 调整 y 轴坐标
# plt.xlim(-2,104)  ### 调整 x 轴坐标
ax=plt.gca()
plt.legend(loc='lower right')
plt.grid()
plt.savefig(data_path+fname+'-test-acc.eps',dpi=300)  ### 保存图片
plt.show()
plt.close()



##########  绘制 Test Loss 图像  ##########
plt.figure(num=4,dpi=300)
for alg in algo_nme:
    x = np.arange(0,len(test_loss_dict[alg][0]))+1
    y = np.mean(test_loss_dict[alg],0)
    y_smooth = uniform_filter1d(y, size=5)
    label = alg.replace("dsam-", " $\\eta$=")
    plt.plot(x, y_smooth, label=label, linewidth=1.5)
    

plt.xlabel("Epochs")
plt.ylabel("Test Loss")
ax=plt.gca();  ### 获得坐标轴的句柄  
ax.spines['bottom'].set_linewidth(2);  ### 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ### 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);  ### 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);  ### 设置上部坐标轴的粗细

# plt.ylim(0.3,2.2)  ### 调整 y 轴坐标
# plt.xlim(-2,104)  ### 调整 x 轴坐标
plt.legend()
plt.grid()
plt.savefig(data_path+fname+'-test-loss.eps',dpi=300)  ### 保存图片
plt.show()
plt.close()
