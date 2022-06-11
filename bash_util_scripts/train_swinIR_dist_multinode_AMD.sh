WORK_DIR=$PWD

#Script for doing all the setup for the app such as installing dependencies
# SETUPSCRIPT=${WORK_DIR}/distributed_run/node_environment_setup.sh

#Runs on each node to setup distributed Python execution
JOBSCRIPT=${WORK_DIR}/run_on_node_AMD.sh

#Training script is executed on each node using job script
APPSCRIPT=$1
JOBDIR=$2   #model dir path
N_NODES=$3

SETUPLOG=/tmp/DistributedSetupLog
RANKSETUPLOG=/tmp/RankSetupLog
RANKERRORLOG=/tmp/RankErrorLog
JOBLOG=/tmp/DistributedJobLogs
JOBERRORLOG=/tmp/DistributedErrorLogs

rm -rf /tmp/Distributed*

#Install Parallel SSH
#echo "Setting up Parallel SSH"
#sudo -H apt-get install pssh

#Run SetupScript on all the nodes
#echo "Running Full Distributed Setup on All Nodes. Check $SETUPLOG"
#parallel-ssh -t 0 -o $SETUPLOG -h /job/mpi-hosts bash $SETUPSCRIPT

#Find appropriate rank for each node
echo "Setting up rank id. Check $RANKSETUPLOG"
#hostname=`hostname -I`
#hostip=`echo $hostname | awk '{print $1}'`

echo "Setting up mpi-hosts. Check $JOBDIR/mpi-hosts"
rm $JOBDIR/mpi-hosts
#cat ~/.ssh/config | grep worker- | cut -d' ' -f2 > $JOBDIR/mpi-hosts

master_ip="node-0"
count=0
sudo -H rm /tmp/host-ranks-master
for j in $(seq $N_NODES):
do
	ip="node-"
	ip+=$count
	host=$ip
    if [ "$count" == 0 ]; then
        rank=0
    else
        rank=$count
    fi
    count=$((count+1))

    echo "$ip $host $rank" >> /tmp/host-ranks-master
    echo "$ip $host $rank"
    echo "$host" >> $JOBDIR/mpi-hosts
done

echo "/tmp/host-ranks-master:"
cat /tmp/host-ranks-master

parallel-scp -o $RANKSETUPLOG -e $RANKERRORLOG -h $JOBDIR/mpi-hosts /tmp/host-ranks-master /tmp/ip-ranks

#Run the actual job script
echo "Running Job Script. Check $JOBLOG and $JOBERRORLOG"
parallel-ssh -x "-tt" -t 0 -o $JOBLOG -e $JOBERRORLOG -h $JOBDIR/mpi-hosts bash $JOBSCRIPT $APPSCRIPT
