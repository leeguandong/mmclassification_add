# mmclassification_add   
已添加squeezenet/ghostnet  

## 如何使用
1.https://blog.csdn.net/u012193416/article/details/124702548     
最新版本的mmcls直接放目录下即可，或者直接pip安装mmcls更省事,但是即便这样，如果你是二分类，还是需要对mmcls做一些更改   
2.add的目录还是要去除openmm系列的学术味，没有必要维护那么多的旨在学术评测的configs。
3.本人的环境一般以多态1080Ti为主，和mm系列的资源不同，一些参数的调整也相对固化。      
4.warmup_iter计算，一般warmup5个epoch左右，数据总量/bs=每个epoch的数据量，一个iter就是一个bs，因此5个epoch就是5*每个cpoch的数据量就是war_iter个数     
5.目前bs在schedules中对应不起来，相应的可能存在lr的需要修改的东西，       

6.常用的算法多配置，摸索自己的成功之道，沉淀自己的经验     
7.多卡训练    
python -m torch.distributed.launch   --nproc_per_node=2   --nnodes=1 --node_rank=0     --master_addr=localhost   --master_port=22222 train.py     






## 比赛经验
1.找到一个好的baseline -> res2net50     
2.

## 辅助工具使用
python analyze_logs.py plot_curve /home/ivms/local_disk/mmclassification-master/tools/results_resnetv1d101_8xb32_in1k/20220513_175100.log.json --keys accuracy_top-1 --out acc.png

python analyze_logs.py plot_curve /home/ivms/local_disk/mmclassification-master/tools/results_resnetv1d101_8xb32_in1k/20220513_175100.log.json

## 比赛
#### 1.CVPR2022 Biometrics Workshop - Image Forgery Detection Challenge
phase1_valset2: 60/674    
phase2_testset: 54

