import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# MAML results paths
maml_path_1 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/maml_trainer/progress.csv'
maml_path_2 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/maml_trainer_1/progress.csv'
maml_path_3 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/maml_trainer_2/progress.csv'
maml_path_4 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/maml_trainer_3/progress.csv'

# PEARL results paths
pearl_path_1 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/pearl_trainer/progress.csv'
pearl_path_2 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/pearl_trainer_1/progress.csv'
pearl_path_3 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/pearl_trainer_2/progress.csv'
pearl_path_4 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/pearl_trainer_3/progress.csv'
pearl_path_5 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/pearl_trainer_4/progress.csv'

# RL2 results paths
rl2_path_1 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/rl2_trainer/progress.csv'
rl2_path_2 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/rl2_trainer_1/progress.csv'
rl2_path_3 = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/experiment/rl2_trainer_2/progress.csv'

# PPO results paths
ppo_path_feeding = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/PPO_assistive_gym_FeedingSawyer-v1/progress.csv'
ppo_path_drinking = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/PPO_assistive_gym_DrinkingSawyer-v1/progress.csv'
ppo_path_bedbathing = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/PPO_assistive_gym_BedBathingSawyer-v1/progress.csv'
ppo_path_scratchitch = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/PPO_assistive_gym_ScratchItchSawyer-v1/progress.csv'
ppo_path_armmanipulation = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/PPO_assistive_gym_ArmManipulationSawyer-v1/progress.csv'

# SAC results paths
sac_path_feeding = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/SAC_assistive_gym_FeedingSawyer-v1/progress.csv'
sac_path_drinking = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/SAC_assistive_gym_DrinkingSawyer-v1/progress.csv'
sac_path_bedbathing = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/SAC_assistive_gym_BedBathingSawyer-v1/progress.csv'
sac_path_scratchitch = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/SAC_assistive_gym_ScratchItchSawyer-v1/progress.csv'
sac_path_armmanipulation = '/home/hisham246/uwaterloo/CS885-MetaRL-AssistiveRobotics/ray_results/SAC_assistive_gym_ArmManipulationSawyer-v1/progress.csv'

# MAML results
maml_df_1 = pd.read_csv(maml_path_1)
maml_df_2 = pd.read_csv(maml_path_2)
maml_df_3 = pd.read_csv(maml_path_3)
maml_df_4 = pd.read_csv(maml_path_4)

# PEARL results
pearl_df_1 = pd.read_csv(pearl_path_1)
pearl_df_2 = pd.read_csv(pearl_path_2)
pearl_df_3 = pd.read_csv(pearl_path_3)
pearl_df_4 = pd.read_csv(pearl_path_4)
pearl_df_5 = pd.read_csv(pearl_path_5)

# RL2 results
rl2_df_1 = pd.read_csv(rl2_path_1)
rl2_df_2 = pd.read_csv(rl2_path_2)
rl2_df_3 = pd.read_csv(rl2_path_3)

# PPO results
ppo_df_feeding = pd.read_csv(ppo_path_feeding)
ppo_df_drinking = pd.read_csv(ppo_path_drinking)
ppo_df_bedbathing = pd.read_csv(ppo_path_bedbathing)
ppo_df_scratchitch = pd.read_csv(ppo_path_scratchitch)
ppo_df_armmanipulation = pd.read_csv(ppo_path_armmanipulation)

# SAC results
sac_df_feeding = pd.read_csv(sac_path_feeding)
sac_df_drinking = pd.read_csv(sac_path_drinking)
sac_df_bedbathing = pd.read_csv(sac_path_bedbathing)
sac_df_scratchitch = pd.read_csv(sac_path_scratchitch)
sac_df_armmanipulation = pd.read_csv(sac_path_armmanipulation)

# MAML
maml_train_epoch_1 = pd.to_numeric(maml_df_1['Average/Iteration'], errors='coerce')
maml_train_reward_1 = pd.to_numeric(maml_df_1['Average/AverageReturn'], errors='coerce')
maml_train_std_1 = pd.to_numeric(maml_df_1['Average/StdReturn'], errors='coerce')
maml_test_epoch_1 = pd.to_numeric(maml_df_1['Average/Iteration'], errors='coerce')
maml_test_reward_1 = pd.to_numeric(maml_df_1['Average/AverageReturn'], errors='coerce')
maml_test_std_1 = pd.to_numeric(maml_df_1['Average/StdReturn'], errors='coerce')
maml_env_1 = pd.to_numeric(maml_df_1['TotalEnvSteps'], errors='coerce')

maml_train_epoch_2 = pd.to_numeric(maml_df_2['Average/Iteration'], errors='coerce')
maml_train_reward_2 = pd.to_numeric(maml_df_2['Average/AverageReturn'], errors='coerce')
maml_train_std_2 = pd.to_numeric(maml_df_2['Average/StdReturn'], errors='coerce')
maml_test_epoch_2 = pd.to_numeric(maml_df_2['Average/Iteration'], errors='coerce')
maml_test_reward_2 = pd.to_numeric(maml_df_2['Average/AverageReturn'], errors='coerce')
maml_test_std_2 = pd.to_numeric(maml_df_2['Average/StdReturn'], errors='coerce')
maml_env_2 = pd.to_numeric(maml_df_2['TotalEnvSteps'], errors='coerce')

maml_train_epoch_3 = pd.to_numeric(maml_df_3['Average/Iteration'], errors='coerce')
maml_train_reward_3 = pd.to_numeric(maml_df_3['Average/AverageReturn'], errors='coerce')
maml_train_std_3 = pd.to_numeric(maml_df_3['Average/StdReturn'], errors='coerce')
maml_test_epoch_3 = pd.to_numeric(maml_df_3['Average/Iteration'], errors='coerce')
maml_test_reward_3 = pd.to_numeric(maml_df_3['Average/AverageReturn'], errors='coerce')
maml_test_std_3 = pd.to_numeric(maml_df_3['Average/StdReturn'], errors='coerce')
maml_env_3 = pd.to_numeric(maml_df_3['TotalEnvSteps'], errors='coerce')

maml_train_epoch_4 = pd.to_numeric(maml_df_4['Average/Iteration'], errors='coerce')
maml_train_reward_4 = pd.to_numeric(maml_df_4['Average/AverageReturn'], errors='coerce')
maml_train_std_4 = pd.to_numeric(maml_df_4['Average/StdReturn'], errors='coerce')
maml_test_epoch_4 = pd.to_numeric(maml_df_4['Average/Iteration'], errors='coerce')
maml_test_reward_4 = pd.to_numeric(maml_df_4['Average/AverageReturn'], errors='coerce')
maml_test_std_4 = pd.to_numeric(maml_df_4['Average/StdReturn'], errors='coerce')
maml_env_3 = pd.to_numeric(maml_df_4['TotalEnvSteps'], errors='coerce')

# PEARL
pearl_epoch_1 = pd.to_numeric(pearl_df_1['MetaTest/Average/Iteration'], errors='coerce')
pearl_reward_1 = pd.to_numeric(pearl_df_1['MetaTest/Average/AverageReturn'], errors='coerce')
pearl_std_1 = pd.to_numeric(pearl_df_1['MetaTest/Average/StdReturn'], errors='coerce')
pearl_env_1 = pd.to_numeric(pearl_df_1['TotalEnvSteps'], errors='coerce')

pearl_epoch_2 = pd.to_numeric(pearl_df_2['MetaTest/Average/Iteration'], errors='coerce')
pearl_reward_2 = pd.to_numeric(pearl_df_2['MetaTest/Average/AverageReturn'], errors='coerce')
pearl_std_2 = pd.to_numeric(pearl_df_2['MetaTest/Average/StdReturn'], errors='coerce')
pearl_env_2 = pd.to_numeric(pearl_df_2['TotalEnvSteps'], errors='coerce')

pearl_epoch_3 = pd.to_numeric(pearl_df_3['MetaTest/Average/Iteration'], errors='coerce')
pearl_reward_3 = pd.to_numeric(pearl_df_3['MetaTest/Average/AverageReturn'], errors='coerce')
pearl_std_3 = pd.to_numeric(pearl_df_3['MetaTest/Average/StdReturn'], errors='coerce')
pearl_env_3 = pd.to_numeric(pearl_df_3['TotalEnvSteps'], errors='coerce')

pearl_epoch_4 = pd.to_numeric(pearl_df_4['MetaTest/Average/Iteration'], errors='coerce')
pearl_reward_4 = pd.to_numeric(pearl_df_4['MetaTest/Average/AverageReturn'], errors='coerce')
pearl_std_4 = pd.to_numeric(pearl_df_4['MetaTest/Average/StdReturn'], errors='coerce')
pearl_env_4 = pd.to_numeric(pearl_df_4['TotalEnvSteps'], errors='coerce')

pearl_epoch_5 = pd.to_numeric(pearl_df_5['MetaTest/Average/Iteration'], errors='coerce')
pearl_reward_5 = pd.to_numeric(pearl_df_5['MetaTest/Average/AverageReturn'], errors='coerce')
pearl_std_5 = pd.to_numeric(pearl_df_5['MetaTest/Average/StdReturn'], errors='coerce')
pearl_env_5 = pd.to_numeric(pearl_df_5['TotalEnvSteps'], errors='coerce')

# RL2
rl2_epoch_1 = pd.to_numeric(rl2_df_1['Average/Iteration'], errors='coerce')
rl2_reward_1 = pd.to_numeric(rl2_df_1['Average/AverageReturn'], errors='coerce')
rl2_std_1 = pd.to_numeric(rl2_df_1['Average/StdReturn'], errors='coerce')
rl2_env_1 = pd.to_numeric(rl2_df_1['TotalEnvSteps'], errors='coerce')

rl2_epoch_2 = pd.to_numeric(rl2_df_2['Average/Iteration'], errors='coerce')
rl2_reward_2 = pd.to_numeric(rl2_df_2['Average/AverageReturn'], errors='coerce')
rl2_std_2 = pd.to_numeric(rl2_df_2['Average/StdReturn'], errors='coerce')
rl2_env_2 = pd.to_numeric(rl2_df_2['TotalEnvSteps'], errors='coerce')

rl2_epoch_3 = pd.to_numeric(rl2_df_3['Average/Iteration'], errors='coerce')
rl2_reward_3 = pd.to_numeric(rl2_df_3['Average/AverageReturn'], errors='coerce')
rl2_std_3 = pd.to_numeric(rl2_df_3['Average/StdReturn'], errors='coerce')
rl2_env_3 = pd.to_numeric(rl2_df_3['TotalEnvSteps'], errors='coerce')

# PPO
ppo_epoch_feeding = pd.to_numeric(ppo_df_feeding['training_iteration'], errors='coerce')
ppo_reward_feeding = pd.to_numeric(ppo_df_feeding['episode_reward_mean'], errors='coerce')
ppo_max_feeding = pd.to_numeric(ppo_df_feeding['episode_reward_max'], errors='coerce')
ppo_min_feeding = pd.to_numeric(ppo_df_feeding['episode_reward_min'], errors='coerce')
ppo_env_feeding = pd.to_numeric(ppo_df_feeding['timesteps_total'], errors='coerce')

ppo_epoch_drinking = pd.to_numeric(ppo_df_drinking['training_iteration'], errors='coerce')
ppo_reward_drinking = pd.to_numeric(ppo_df_drinking['episode_reward_mean'], errors='coerce')
ppo_max_drinking = pd.to_numeric(ppo_df_drinking['episode_reward_max'], errors='coerce')
ppo_min_drinking = pd.to_numeric(ppo_df_drinking['episode_reward_min'], errors='coerce')
ppo_env_drinking = pd.to_numeric(ppo_df_drinking['timesteps_total'], errors='coerce')

ppo_epoch_bedbathing = pd.to_numeric(ppo_df_bedbathing['training_iteration'], errors='coerce')
ppo_reward_bedbathing = pd.to_numeric(ppo_df_bedbathing['episode_reward_mean'], errors='coerce')
ppo_max_bedbathing = pd.to_numeric(ppo_df_bedbathing['episode_reward_max'], errors='coerce')
ppo_min_bedbathing = pd.to_numeric(ppo_df_bedbathing['episode_reward_min'], errors='coerce')
ppo_env_bedbathing = pd.to_numeric(ppo_df_bedbathing['timesteps_total'], errors='coerce')

ppo_epoch_scratchitch = pd.to_numeric(ppo_df_scratchitch['training_iteration'], errors='coerce')
ppo_reward_scratchitch = pd.to_numeric(ppo_df_scratchitch['episode_reward_mean'], errors='coerce')
ppo_max_scratchitch = pd.to_numeric(ppo_df_scratchitch['episode_reward_max'], errors='coerce')
ppo_min_scratchitch = pd.to_numeric(ppo_df_scratchitch['episode_reward_min'], errors='coerce')
ppo_env_scratchitch = pd.to_numeric(ppo_df_scratchitch['timesteps_total'], errors='coerce')

ppo_epoch_armmanipulation = pd.to_numeric(ppo_df_armmanipulation['training_iteration'], errors='coerce')
ppo_reward_armmanipulation = pd.to_numeric(ppo_df_armmanipulation['episode_reward_mean'], errors='coerce')
ppo_max_armmanipulation = pd.to_numeric(ppo_df_armmanipulation['episode_reward_max'], errors='coerce')
ppo_min_armmanipulation = pd.to_numeric(ppo_df_armmanipulation['episode_reward_min'], errors='coerce')
ppo_env_armmanipulation = pd.to_numeric(ppo_df_armmanipulation['timesteps_total'], errors='coerce')

# SAC
sac_epoch_feeding = pd.to_numeric(sac_df_feeding['training_iteration'], errors='coerce')
sac_reward_feeding = pd.to_numeric(sac_df_feeding['episode_reward_mean'], errors='coerce')
sac_max_feeding = pd.to_numeric(sac_df_feeding['episode_reward_max'], errors='coerce')
sac_min_feeding = pd.to_numeric(sac_df_feeding['episode_reward_min'], errors='coerce')
sac_env_feeding = pd.to_numeric(sac_df_feeding['timesteps_total'], errors='coerce')

sac_epoch_drinking = pd.to_numeric(sac_df_drinking['training_iteration'], errors='coerce')
sac_reward_drinking = pd.to_numeric(sac_df_drinking['episode_reward_mean'], errors='coerce')
sac_max_drinking = pd.to_numeric(sac_df_drinking['episode_reward_max'], errors='coerce')
sac_min_drinking = pd.to_numeric(sac_df_drinking['episode_reward_min'], errors='coerce')
sac_env_drinking = pd.to_numeric(sac_df_drinking['timesteps_total'], errors='coerce')

sac_epoch_bedbathing = pd.to_numeric(sac_df_bedbathing['training_iteration'], errors='coerce')
sac_reward_bedbathing = pd.to_numeric(sac_df_bedbathing['episode_reward_mean'], errors='coerce')
sac_max_bedbathing = pd.to_numeric(sac_df_bedbathing['episode_reward_max'], errors='coerce')
sac_min_bedbathing = pd.to_numeric(sac_df_bedbathing['episode_reward_min'], errors='coerce')
sac_env_bedbathing = pd.to_numeric(sac_df_bedbathing['timesteps_total'], errors='coerce')

sac_epoch_scratchitch = pd.to_numeric(sac_df_scratchitch['training_iteration'], errors='coerce')
sac_reward_scratchitch = pd.to_numeric(sac_df_scratchitch['episode_reward_mean'], errors='coerce')
sac_max_scratchitch = pd.to_numeric(sac_df_scratchitch['episode_reward_max'], errors='coerce')
sac_min_scratchitch = pd.to_numeric(sac_df_scratchitch['episode_reward_min'], errors='coerce')
sac_env_scratchitch = pd.to_numeric(sac_df_scratchitch['timesteps_total'], errors='coerce')

sac_epoch_armmanipulation = pd.to_numeric(sac_df_armmanipulation['training_iteration'], errors='coerce')
sac_reward_armmanipulation = pd.to_numeric(sac_df_armmanipulation['episode_reward_mean'], errors='coerce')
sac_max_armmanipulation = pd.to_numeric(sac_df_armmanipulation['episode_reward_max'], errors='coerce')
sac_min_armmanipulation = pd.to_numeric(sac_df_armmanipulation['episode_reward_min'], errors='coerce')
sac_env_armmanipulation = pd.to_numeric(sac_df_armmanipulation['timesteps_total'], errors='coerce')

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

def feeding():
    plt.figure(figsize=(10, 6))
    # PPO
    plt.plot(ppo_env_feeding, ppo_reward_feeding, label=r'\textbf{PPO}', color='tab:blue')
    plt.fill_between(ppo_env_feeding,
                    ppo_min_feeding,
                    ppo_max_feeding,
                    color='tab:blue', alpha=0.2)

    # SAC
    plt.plot(sac_env_feeding, sac_reward_feeding, label=r'\textbf{SAC}', color='tab:orange')
    plt.fill_between(sac_env_feeding,
                    sac_min_feeding,
                    sac_max_feeding,
                    color='tab:orange', alpha=0.2)
    
    # Axes and formatting
    plt.xlabel(r'\textbf{Environment Steps}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Feeding}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.show()

def drinking():
    plt.figure(figsize=(10, 6))
    # PPO
    plt.plot(ppo_env_drinking, ppo_reward_drinking, label=r'\textbf{PPO}', color='tab:blue')
    plt.fill_between(ppo_env_drinking,
                    ppo_min_drinking,
                    ppo_max_drinking,
                    color='tab:blue', alpha=0.2)

    # SAC
    plt.plot(sac_env_drinking, sac_reward_drinking, label=r'\textbf{SAC}', color='tab:orange')
    plt.fill_between(sac_env_drinking,
                    sac_min_drinking,
                    sac_max_drinking,
                    color='tab:orange', alpha=0.2)
    
    # Axes and formatting
    plt.xlabel(r'\textbf{Environment Steps}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Drinking}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.show()

def bedbathing():
    plt.figure(figsize=(10, 6))
    # PPO
    plt.plot(ppo_env_bedbathing, ppo_reward_bedbathing, label=r'\textbf{PPO}', color='tab:blue')
    plt.fill_between(ppo_env_bedbathing,
                    ppo_min_bedbathing,
                    ppo_max_bedbathing,
                    color='tab:blue', alpha=0.2)

    # SAC
    plt.plot(sac_env_bedbathing, sac_reward_bedbathing, label=r'\textbf{SAC}', color='tab:orange')
    plt.fill_between(sac_env_bedbathing,
                    sac_min_bedbathing,
                    sac_max_bedbathing,
                    color='tab:orange', alpha=0.2)
    
    # Axes and formatting
    plt.xlabel(r'\textbf{Environment Steps}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Bed Bathing}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.show()

def scratchitch():
    plt.figure(figsize=(10, 6))
    # PPO
    plt.plot(ppo_env_scratchitch, ppo_reward_scratchitch, label=r'\textbf{PPO}', color='tab:blue')
    plt.fill_between(ppo_env_scratchitch,
                    ppo_min_scratchitch,
                    ppo_max_scratchitch,
                    color='tab:blue', alpha=0.2)

    # SAC
    plt.plot(sac_env_scratchitch, sac_reward_scratchitch, label=r'\textbf{SAC}', color='tab:orange')
    plt.fill_between(sac_env_scratchitch,
                    sac_min_scratchitch,
                    sac_max_scratchitch,
                    color='tab:orange', alpha=0.2)
    
    # Axes and formatting
    plt.xlabel(r'\textbf{Environment Steps}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Scratch Itch}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.show()

def armmanipulation():
    plt.figure(figsize=(10, 6))
    # PPO
    plt.plot(ppo_env_armmanipulation, ppo_reward_armmanipulation, label=r'\textbf{PPO}', color='tab:blue')
    plt.fill_between(ppo_env_armmanipulation,
                    ppo_min_armmanipulation,
                    ppo_max_armmanipulation,
                    color='tab:blue', alpha=0.2)

    # SAC
    plt.plot(sac_env_armmanipulation, sac_reward_armmanipulation, label=r'\textbf{SAC}', color='tab:orange')
    plt.fill_between(sac_env_armmanipulation,
                    sac_min_armmanipulation,
                    sac_max_armmanipulation,
                    color='tab:orange', alpha=0.2)
    
    # Axes and formatting
    plt.xlabel(r'\textbf{Environment Steps}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Arm Manipulation}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.show()


def two_task_adaptation():
    plt.figure(figsize=(10, 6))
    # MAML
    plt.plot(maml_test_epoch_1, maml_test_reward_1, label=r'\textbf{MAML}', color='tab:blue')
    plt.fill_between(maml_test_epoch_1,
                    maml_test_reward_1 - maml_test_std_1,
                    maml_test_reward_1 + maml_test_std_1,
                    color='tab:blue', alpha=0.2)

    # PEARL
    plt.plot(pearl_epoch_1, pearl_reward_1, label=r'\textbf{PEARL}', color='tab:orange')
    plt.fill_between(pearl_epoch_1,
                    pearl_reward_1 - pearl_std_1,
                    pearl_reward_1 + pearl_std_1,
                    color='tab:orange', alpha=0.2)

    # RL2
    plt.plot(rl2_epoch_2, rl2_reward_2, label=r'\textbf{RL$^2$}', color='tab:green')
    plt.fill_between(rl2_epoch_2,
                    rl2_reward_2 - rl2_std_2,
                    rl2_reward_2 + rl2_std_2,
                    color='tab:green', alpha=0.2)

    # Axes and formatting
    plt.xlabel(r'\textbf{Epoch}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Adaptation Across Feeding and Drinking Tasks}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/metarl_feeding_drinking.pdf', format='pdf', bbox_inches='tight', dpi=300)

def three_task_adaptation():
    plt.figure(figsize=(10, 6))
    # MAML
    plt.plot(maml_test_epoch_2, maml_test_reward_2, label=r'\textbf{MAML}', color='tab:blue')
    plt.fill_between(maml_test_epoch_2,
                    maml_test_reward_2 - maml_test_std_2,
                    maml_test_reward_2 + maml_test_std_2,
                    color='tab:blue', alpha=0.2)

    # PEARL
    plt.plot(pearl_epoch_3[:300], pearl_reward_3[:300], label=r'\textbf{PEARL}', color='tab:orange')
    plt.fill_between(pearl_epoch_3[:300],
                    pearl_reward_3[:300] - pearl_std_3[:300],
                    pearl_reward_3[:300] + pearl_std_3[:300],
                    color='tab:orange', alpha=0.2)

    # Axes and formatting
    plt.xlabel(r'\textbf{Epoch}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Adaptation Across Itch Scratching, Bed Bathing, and Arm Manipulation Tasks}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/metarl_hygiene.pdf', format='pdf', bbox_inches='tight', dpi=300)

def five_task_adaptation():
    plt.figure(figsize=(10, 6))
    # MAML
    plt.plot(maml_test_epoch_4, maml_test_reward_4, label=r'\textbf{MAML}', color='tab:blue')
    plt.fill_between(maml_test_epoch_4,
                    maml_test_reward_4 - maml_test_std_4,
                    maml_test_reward_4 + maml_test_std_4,
                    color='tab:blue', alpha=0.2)

    # PEARL
    plt.plot(pearl_epoch_5, pearl_reward_5, label=r'\textbf{PEARL}', color='tab:orange')
    plt.fill_between(pearl_epoch_5,
                    pearl_reward_5 - pearl_std_5,
                    pearl_reward_5 + pearl_std_5,
                    color='tab:orange', alpha=0.2)

    # Axes and formatting
    plt.xlabel(r'\textbf{Epoch}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Adaptation Across All Assistive Tasks}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/metarl_all.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Test across different robots
def robot_type_adaptation():
    plt.figure(figsize=(10, 6))

    # MAML
    plt.plot(maml_test_epoch_3, maml_test_reward_3, label=r'\textbf{MAML}', color='tab:blue')
    plt.fill_between(maml_test_epoch_3,
                    maml_test_reward_3 - maml_test_std_3,
                    maml_test_reward_3 + maml_test_std_3,
                    color='tab:blue', alpha=0.2)

    # PEARL
    plt.plot(pearl_epoch_4[:300], pearl_reward_4[:300], label=r'\textbf{PEARL}', color='tab:orange')
    plt.fill_between(pearl_epoch_4[:300],
                    pearl_reward_4[:300] - pearl_std_4[:300],
                    pearl_reward_4[:300] + pearl_std_4[:300],
                    color='tab:orange', alpha=0.2)

    # RL2
    plt.plot(rl2_epoch_3[:300], rl2_reward_3[:300], label=r'\textbf{RL$^2$}', color='tab:green')
    plt.fill_between(rl2_epoch_3[:300],
                    rl2_reward_3[:300] - rl2_std_3[:300],
                    rl2_reward_3[:300] + rl2_std_3[:300],
                    color='tab:green', alpha=0.2)

    # Axes and formatting
    plt.xlabel(r'\textbf{Epoch}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Adaptation Across Different Robots}', fontsize=18)
    plt.legend(loc='upper left', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/metarl_robot_type.pdf', format='pdf', bbox_inches='tight', dpi=300)


def two_task_comparison():
    plt.figure(figsize=(10, 6))
    # MAML
    plt.plot(maml_test_epoch_1, maml_test_reward_1, label=r'\textbf{MAML}', color='tab:blue')
    plt.fill_between(maml_test_epoch_1,
                    maml_test_reward_1 - maml_test_std_1,
                    maml_test_reward_1 + maml_test_std_1,
                    color='tab:blue', alpha=0.2)

    # PEARL
    plt.plot(pearl_epoch_1, pearl_reward_1, label=r'\textbf{PEARL}', color='tab:orange')
    plt.fill_between(pearl_epoch_1,
                    pearl_reward_1 - pearl_std_1,
                    pearl_reward_1 + pearl_std_1,
                    color='tab:orange', alpha=0.2)

    # RL2
    plt.plot(rl2_epoch_2, rl2_reward_2, label=r'\textbf{RL$^2$}', color='tab:green')
    plt.fill_between(rl2_epoch_2,
                    rl2_reward_2 - rl2_std_2,
                    rl2_reward_2 + rl2_std_2,
                    color='tab:green', alpha=0.2)
    
    # PPO
    plt.plot(rl2_epoch_2, rl2_reward_2, label=r'\textbf{RL$^2$}', color='tab:green')
    plt.fill_between(rl2_epoch_2,
                    rl2_reward_2 - rl2_std_2,
                    rl2_reward_2 + rl2_std_2,
                    color='tab:green', alpha=0.2)

    # Axes and formatting
    plt.xlabel(r'\textbf{Epoch}')
    plt.ylabel(r'\textbf{Average Return}')
    plt.title(r'\textbf{Adaptation Across Feeding and Drinking Tasks}', fontsize=18)
    plt.legend(loc='lower right', facecolor='white', framealpha=1.0)
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/metarl_feeding_drinking.pdf', format='pdf', bbox_inches='tight', dpi=300)


# feeding()
# drinking()
# bedbathing()
# scratchitch()
# armmanipulation()

# two_task_adaptation()
# three_task_adaptation()
# five_task_adaptation()
# robot_type_adaptation()