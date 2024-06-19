
MODEL_NAME="CompVis/stable-diffusion-v1-4"

MY_CMD="python classifier.py --mode train --arch resnet50 --data_dir {data_path} --test_dir {optional} --split_val --val_epoch 1 --batch_size 128 --learning_rate 0.001 --epochs 80 --save_name {save_name}"

MY_CMD="python classifier.py --mode test --arch resnet50 --data_dir None --test_dir {test_data} --split_val --val_epoch 1 --batch_size 128 --learning_rate 0.001 --epochs 50 --save_name copyright --to_test_method {ckpt}"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='0' $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
