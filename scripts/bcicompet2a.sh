#!/bin/bash

sub_arr_1=(0)
sub_arr_2=(1)
sub_arr_3=(2)
sub_arr_4=(3)
sub_arr_5=(4)
sub_arr_6=(5)
sub_arr_7=(6)
sub_arr_8=(7)
sub_arr_9=(8)
fold_arr=(0 1 2 3 4)

function train_process()
{
    sub_arr=("${!1}")
    fold_arr=("${!2}")
    gpu_num=$3

    for i in ${sub_arr[@]}; do
        for j in ${fold_arr[@]}; do
            python training.py --subject_num=$i --fold_num=$j --gpu_num=$gpu_num --config_name="bcicompet2a_config"
        done
    done
}

train_process sub_arr_1[@] fold_arr[@] 0 &
train_process sub_arr_2[@] fold_arr[@] 0 &
train_process sub_arr_3[@] fold_arr[@] 0 &
train_process sub_arr_4[@] fold_arr[@] 1 &
train_process sub_arr_5[@] fold_arr[@] 1 &
train_process sub_arr_6[@] fold_arr[@] 1 &
train_process sub_arr_7[@] fold_arr[@] 2 &
train_process sub_arr_8[@] fold_arr[@] 2 &
train_process sub_arr_9[@] fold_arr[@] 2