set -e

hostname
whoami

# Create redirect on master node
mkdir -p /tmp/.ssh && rm -f /tmp/.ssh/config && ln -s ~/.ssh/config /tmp/.ssh/config && echo "/tmp/.ssh/config created" && ls -l /tmp/.ssh
# mkdir -p /opt && sudo rm -f /opt/hostfile && sudo ln -s /job/hostfile /opt/hostfile && cat /opt/hostfile

# Create redirect for all workers
for i in $(seq 0 $(( ${DLWS_NUM_WORKER} - 1 ))); do
    echo "Creating redirect for worker ${i}"
    ssh worker-${i} "mkdir -p /tmp/.ssh && rm -f /tmp/.ssh/config && ln -s ~/.ssh/config /tmp/.ssh/config && echo "/tmp/.ssh/config created" && ls -l /tmp/.ssh"
    # ssh worker-${i} "mkdir -p /opt && sudo rm -f /opt/hostfile && sudo ln -s /job/hostfile /opt/hostfile && cat /opt/hostfile"
done

