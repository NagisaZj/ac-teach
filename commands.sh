CUDA_VISIBLE_DEVICES=6 python scripts/run.py --config cfg/pick_place/experiments/efficiency/partial_complete_suboptimal/train_ours.yaml

CUDA_VISIBLE_DEVICES=1 python scripts/run.py --config cfg/metaworld/experiments/efficiency/partial_complete_suboptimal/train_ours.yaml

#0 1 bin
#2 3 pp
#4 5 6 7 pp one teacher
# 8 9 pp one teacher test (behavior teacher only)
# 10 11 pp one teacher test add random (behavior teacher only)
# 12 13 behavior teacher only
# 14 15 original
# 16 17 pp original fix goal
# 18 19 reach original fix goal
# 20 21 reach original random goal
# 22 23 reach original random goal push source
# 24 25 push original random goal push source
# 29 push original single goal push source
# 30 push original random goal ddpg
# 30 push original fix goal ddpg