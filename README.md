[📘使用文档]() |
[🛠安装教程]() |
[👀模型库]() |
[🆕更新日志]() |
[🚀进行中的项目]() |
[🤔报告问题]()

</div>

 ## 简介
mmclassification_add是对mmclassification的补充项目，和众多add系列项目一样，它旨在添加一些mm中未收录的额外的方法，但是又按照mm系列的框架来看，为了复用
包括runner在内的众多特性，此外mmclassification的更新，只需要等价替换mmclas目录即可，configs按照现实工程进行配置，没有必要维护那么多的旨在学术评测的configs。  

<details open>
<summary>主要特性</summary>

- **便捷**     
    不需要改动原始mmclass的代码
    

</details>

## 如何使用
- **调试**  
最新版本的mmcls直接放目录下即可，或者直接pip安装mmcls更省事,但是即便这样，如果你是二分类，还是需要对mmcls做一些[更改](https://blog.csdn.net/u012193416/article/details/124702548 )      

- **训练资源**     
训练环境一般以多态1080Ti为主，和mm系列的资源不同，一些参数的调整也相对固化。      

- **warmup_iter**    
warmup_iter计算，一般warmup5个epoch左右，数据总量/bs=每个epoch的数据量，一个iter就是一个bs，因此5个epoch就是5*每个cpoch的数据量就是war_iter个数     

- **bs**
目前bs在schedules中对应不起来，相应的可能存在lr的需要修改的东西       
    
- **多卡训练**     
python -m torch.distributed.launch   --nproc_per_node=2   --nnodes=1 --node_rank=0     --master_addr=localhost   --master_port=22222 train.py    
     
- **onnx**
对外提供接口的前向全部切换成onnx    

- **线上训练报错**      
ModuleNotFoundError: No module named '_lzma'??     
sudo yum install xz-devel -y    
sudo yum install python-backports-lzma -y    
将requirements下的lzma.py cp到/usr/local/python3/lib/python3.6下，并安装pip install backports.lzma    

- **线上验证**


- **结果打印**
plt打印中文有乱码
mmcls/core/visualization/image.py中字体设置207行将monospace换成SimHei      

- **mmcv版本**
mmcv_full-1.4.4-cp36-cp36m-manylinux1_x86_64.whl
torch-1.7cu92
torchvision==0.8.0






## mmclass_add中添加的算法
- ✅ [squeezenet](https://arxiv.org/abs/1712.01026)
- ✅ [ghostnet](https://blog.csdn.net/u012193416/article/details/125716540?spm=1001.2014.3001.5501)


## 经验
- **backbone**     
好的baseline -> res2net50     

## 部署模块 deploy



## 辅助工具使用
- **loss/acc**    
python analyze_logs.py plot_curve /home/ivms/local_disk/mmclassification-master/tools/results_resnetv1d101_8xb32_in1k/20220513_175100.log.json --keys accuracy_top-1 --out acc.png     
python analyze_logs.py plot_curve /home/ivms/local_disk/mmclassification-master/tools/results_resnetv1d101_8xb32_in1k/20220513_175100.log.json

## 比赛及项目
#### 1.CVPR2022 Biometrics Workshop - Image Forgery Detection Challenge
phase1_valset2: 60/674    
phase2_testset: 54

#### 2.科大讯飞 LED色彩和均匀性检测


#### 3.家装家居素材风格识别



