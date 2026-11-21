#!/bin/bash
# Submit the gen-floor job chains from ~/vw-genfloor (run on the Delta login node).
set -euo pipefail
cd ~/vw-genfloor
mkdir -p logs
SYN=/work/hdd/bjeg/meme-pose/work/synthetic
GPU=gpuA40x4,gpuA100x4,gpuA100x8,gpuH200x8
PRE='umask 077 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate memepose && cd ~/vw-genfloor && G="python -m memepose.generate.run --config configs/delta.yaml"'
# every label except shrek_side_eye (editors pull the gaze back to the camera; real data only)
RETRY=arms_raised,confused_tilt,hands_on_head,holding_laugh,no_meme,pinch_grin,point_at_self,point_sideways,pure_cinema,sleepy_squint,timeout
cpu() {  # cpu <name> <time> [dependency] <command>
  local name=$1 t=$2 dep=$3 cmd=$4
  sbatch --parsable ${dep:+--dependency=$dep} -p cpu --account=bjeg-delta-cpu --cpus-per-task=8 --mem=32G \
    -t "$t" -J "$name" -o logs/%x-%j.out -e logs/%x-%j.err --wrap "$PRE && $cmd"
}
gen() {  # gen <run> <shards> <time> [dependency]
  sbatch --parsable ${4:+--dependency=$4} -p $GPU -t "$3" --array=0-$(($2 - 1)) \
    --export=ALL,RUN=$1,BACKEND=klein4b slurm/generate.slurm
}
m2() {  # m2 <tag> <dependency> <extra> [time]
  sbatch --parsable --dependency=$2 -p $GPU -t "${4:-03:00:00}" --export="ALL,TAG=$1,EXTRA=$3" slurm/m2.slurm
}

# 1. calibration of the check thresholds (numbers only)
cal=$(cpu memepose-calibrate 03:00:00 "" "python -m memepose.generate.calibrate --config configs/delta.yaml")

# 2. batch1b: recheck batch1 with the fixed checks, retry rejects once, train syn/ctl
b1r=$(cpu memepose-recheck 04:00:00 "" "\$G recheck --name batch1b && \$G merge --name batch1b && \$G retry --name batch1b --labels $RETRY")
b1g=$(gen batch1b 8 01:00:00 afterok:$b1r)
b1m=$(cpu memepose-merge 00:30:00 afterany:$b1g "\$G merge --name batch1b")
b1s=$(m2 syn_b1b afterok:$b1m "--synthetic-images $SYN/batch1b/synthetic_frames.csv")
b1c=$(m2 ctl_b1b afterok:$b1m "--synthetic-control --synthetic-images $SYN/batch1b/synthetic_frames.csv")

# 3. batch2 (the large run): plan made on the login node beforehand; 16 short shards
#    (short jobs backfill on the busy queue), merge + one retry round, then syn/ctl
b2g=$(gen batch2 16 01:30:00)
b2r=$(cpu memepose-retry 00:30:00 afterany:$b2g "\$G merge --name batch2 && \$G retry --name batch2 --labels $RETRY")
b2g2=$(gen batch2 16 01:00:00 afterok:$b2r)  # same shard count as round 1 (edit resumes per shard file); also finishes any edit a failed first-round shard left
b2m=$(cpu memepose-merge 00:30:00 afterany:$b2g2 "\$G merge --name batch2")
b2s=$(m2 syn_b2 afterok:$b2m "--synthetic-images $SYN/batch2/synthetic_frames.csv" 05:00:00)
b2c=$(m2 ctl_b2 afterok:$b2m "--synthetic-control --synthetic-images $SYN/batch2/synthetic_frames.csv" 05:00:00)

cat <<EOF
calibrate        $cal
batch1b recheck  $b1r -> gen $b1g -> merge $b1m -> syn_b1b $b1s / ctl_b1b $b1c
batch2 gen       $b2g -> merge+retry $b2r -> gen $b2g2 -> merge $b2m -> syn_b2 $b2s / ctl_b2 $b2c
EOF
