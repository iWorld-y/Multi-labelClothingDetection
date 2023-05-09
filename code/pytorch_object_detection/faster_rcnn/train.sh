python train_res50_fpn.py --data-path "/root/autodl-tmp/" --epochs 2 ; \
mygit "测试训练结束" ; \
python train_res50_fpn.py --data-path "/root/autodl-tmp/" --epochs 300 --batch_size 22  && mygit "Faster RCNN 第一次训练结束，epochs:300"; /usr/bin/shutdown