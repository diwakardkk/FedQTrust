#!/usr/bin/env bash
# Start or resume FedQTrust. Safe to run repeatedly (cron every 10 min + @reboot).
# "touch STOP" pauses it; "rm STOP" re-enables it. gpus.conf lists one GPU ID per worker.
cd "$(dirname "$0")" || exit 1
export PATH=/usr/local/bin:/usr/bin:/bin CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=.venv/bin/python
[ -f STOP ] && exit 0
S=$($PY server.py status 2>/dev/null)
echo "$S" | grep -q '"supervisor_active": true' && exit 0
if echo "$S" | grep -q '"training_complete": true' && echo "$S" | grep -qE '"state": "(COMPLETE|INCOMPLETE)"'; then exit 0; fi
if [ ! -s gpus.conf ]; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
  awk -F', ' '$2<50 && ($4-$3)>6000 { n = ($2<10) ? 2 : 1; for(i=0;i<n;i++) print $1 }' | head -6 | paste -sd, > gpus.conf
fi
G=$(cat gpus.conf); [ -z "$G" ] && { echo "$(date) no free GPU, will retry"; exit 0; }
echo "$(date) starting/resuming on GPUs: $G"
CUDA_VISIBLE_DEVICES=${G%%,*} $PY run.py preflight --config settings.json > /dev/null || { echo "$(date) preflight failed"; exit 1; }
$PY server.py start --gpus "$G"
