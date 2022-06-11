#These variables need to be defined for running distributed pytorch
model_path=$1
input_dir=$2
scale=$3

# set this path
export PYTHONPATH=$PYTHONPATH:/home/saghotra/git

python /home/saghotra/git/KAIR/main_test_swinir.py \
        --task real_sr \
        --model_path $model_path \
        --folder_lq $input_dir \
        --scale $scale \
        --large_model \
#       --tile 800


# Use --tile argument when the image size is too large for
# a GPU memeory
