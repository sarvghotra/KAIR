#These variables need to be defined for running distributed pytorch
numnodes=$1
rank=$2
masterip=$3

num_gpus_per_node=8

CODE_PARENT_DIR="$PWD/../.."
export PYTHONPATH=$PYTHONPATH:$CODE_PARENT_DIR

#Kill Currently Running Jobs
sudo -H pkill python

NCCL_SOCKET_IFNAME=^lo,docker0,veth NCCL_DEBUG=WARN \
NCCL_TREE_THRESHOLD=0 \
python -m torch.distributed.launch \
--nproc_per_node=$num_gpus_per_node \
--nnodes=$numnodes \
--node_rank=$rank \
--master_addr=$masterip \
--master_port=12345 \
$CODE_PARENT_DIR/KAIR/main_train_psnr.py \
--opt /home/saghotra/git/KAIR/options/swinir/train_swinir_large_sr_realworld_x2_gan_fp16.json \
--dist True \
# --resume_crashed_training