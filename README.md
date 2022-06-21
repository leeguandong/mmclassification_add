# mmclassification_add   
已添加squeezenet/ghostnet  

## 如何使用
https://blog.csdn.net/u012193416/article/details/124702548       
最新版本的mmcls直接放目录下即可，或者直接pip安装mmcls更省事,但是即便这样，如果你是二分类，还是需要对mmcls做一些更改     
add的目录还是要去除openmm系列的学术味，没有必要维护那么多的旨在学术评测的configs。    
本人的环境一般以多态1080Ti为主，和mm系列的资源不同，一些参数的调整也相对固化。     


## 辅助工具使用
python analyze_logs.py plot_curve /home/ivms/local_disk/mmclassification-master/tools/results_resnetv1d101_8xb32_in1k/20220513_175100.log.json --keys accuracy_top-1 --out acc.png

python analyze_logs.py plot_curve /home/ivms/local_disk/mmclassification-master/tools/results_resnetv1d101_8xb32_in1k/20220513_175100.log.json

## 比赛
#### 1.CVPR2022 Biometrics Workshop - Image Forgery Detection Challenge
phase1_valset2: 60/674    
phase2_testset: 54

