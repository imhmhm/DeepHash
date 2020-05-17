#!/bin/bash

lr=0.002
q_lambda=0.0001
# subspace_num=4
# n_subcenter=256
dataset=coco # cifar10, nuswide_81, coco
log_dir=tflog

if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

# filename="lr_${lr}_cqlambda_${q_lambda}_subspace_num_${subspace_num}_T_${T}_K_${K}_graph_laplacian_lambda_${gl_lambda}_gl_loss_${gl_loss}_dataset_${dataset}"
# model_file="models/${filename}.npy"
export TF_CPP_MIN_LOG_LEVEL=3
##                                                        lr  output  iter    q_lamb      n_sub   n_center  dataset     gpu    log_dir
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 96    8000    $q_lambda    6    256   coco    0      $log_dir   ../../data/coco

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 64    8000    $q_lambda    4    256   coco    0      $log_dir   ../../data/coco

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 48    8000    $q_lambda    3    256   coco    0      $log_dir   ../../data/coco

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 32    8000    $q_lambda    2    64    coco    0      $log_dir   ../../data/coco

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 96    8000    $q_lambda    6    256   nuswide_81    0      $log_dir   ../../data/nuswide_81

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 64    8000    $q_lambda    4    256   nuswide_81    0      $log_dir   ../../data/nuswide_81

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 48    8000    $q_lambda    3    256   nuswide_81    0      $log_dir   ../../data/nuswide_81

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 32    8000    $q_lambda    2    64    nuswide_81    0      $log_dir   ../../data/nuswide_81

CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 96    8000    $q_lambda    6    256   cifar10    0      $log_dir   ../../data/cifar10
#
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 64    8000    $q_lambda    4    256   cifar10    0      $log_dir   ../../data/cifar10
#
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 48    8000    $q_lambda    3    256   cifar10    0      $log_dir   ../../data/cifar10
#
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     $lr 32    8000    $q_lambda    2    64    cifar10    0      $log_dir   ../../data/cifar10
