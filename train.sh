# Meta RL
python -m assistive_gym.learn_metarl

# PPO_Social_Dempref
# CUDA_VISIBLE_DEVICES=0 python -m assistive_gym.learn_dempref_conditioned  --env "BedBathingSawyerHuman-v1" --seed 1 --setting-name 1 --social --dynamic-future --lr 5e-6 --dempref --continuous-sampling --robot-start 10 --human-start 10 --human-len 4 --robot-len 7 --merge-alpha 0.5

# PPO
# python -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --train --train-timesteps 100000