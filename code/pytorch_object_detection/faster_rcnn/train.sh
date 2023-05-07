python train_res50_fpn.py --data-path "/root/autodl-tmp/" --num-classes 13 --epochs 2 ; \
mygit "测试训练结束" ; \
python train_res50_fpn.py --data-path "/root/autodl-tmp/" --num-classes 13 --epochs 300 ; \
mygit "Faster RCNN 第一次训练结束，epochs:300" ; \
/usr/bin/shutdown