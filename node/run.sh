nuhup python train.py --dataset pubmed --lr 0.001 --gpu 0 --k 8 --eps 0.05 > pubmed-0.001-8-0.05  2>&1
nuhup python train.py --dataset wikics --lr 0.0001 --gpu 0 --k 32 --eps 0.05 > wikics-0.0001-8-0.05  2>&1
nuhup python train.py --dataset amzcom --lr 0.0001 --gpu 0 --k 32 --eps 0.05 > amzcom-0.0001-8-0.05  2>&1
nuhup python train.py --dataset amzphoto --lr 0.0001 --gpu 0 --k 32 --eps 0.05 > amzphoto-0.0001-8-0.05  2>&1