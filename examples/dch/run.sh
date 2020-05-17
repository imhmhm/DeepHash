##===================================
sleep 1h
##===================================
python train_val_script.py \
--data-dir ../../data \
--dataset cifar10 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 12;

python train_val_script.py \
--data-dir ../../data \
--dataset cifar10 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 24;

python train_val_script.py \
--data-dir ../../data \
--dataset cifar10 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 32;

python train_val_script.py \
--data-dir ../../data \
--dataset cifar10 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 48;

##===================================
python train_val_script.py \
--data-dir ../../data \
--dataset nuswide_81 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 12;

python train_val_script.py \
--data-dir ../../data \
--dataset nuswide_81 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 24;

python train_val_script.py \
--data-dir ../../data \
--dataset nuswide_81 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 32;

python train_val_script.py \
--data-dir ../../data \
--dataset nuswide_81 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 48;

##===================================
python train_val_script.py \
--data-dir ../../data \
--dataset coco \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 12;

python train_val_script.py \
--data-dir ../../data \
--dataset coco \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 24;

python train_val_script.py \
--data-dir ../../data \
--dataset coco \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 32;

python train_val_script.py \
--data-dir ../../data \
--dataset coco \
-b 128 \
--lr 0.0005 \
--q-lambda 0.001 \
--iter-num 5000 \
--output-dim 48;

##===================================
python train_val_script.py \
--data-dir ../../data \
--dataset cifar10 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.01 \
--iter-num 5000 \
--output-dim 12;

python train_val_script.py \
--data-dir ../../data \
--dataset cifar10 \
-b 128 \
--lr 0.0005 \
--q-lambda 0.1 \
--iter-num 5000 \
--output-dim 12;

##===================================
