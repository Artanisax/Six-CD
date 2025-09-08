# The indexes of copyrighted characters and objects
# follow their ids in SIX-CD datasets

TEST_DIR=inputs/copyright/v1-4/100
SAVE_NAME=copyright
CKPT=resnet50_copyright_101_71
SAVE_PATH=results/copyright/emcid/100.json

python classifier.py \
    --mode test \
    --arch resnet50 \
    --data_dir None \
    --test_dir $TEST_DIR \
    --split_val \
    --val_epoch 1 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --epochs 50 \
    --save_name copyright \
    --to_test_method $CKPT \
    --save_path $SAVE_PATH