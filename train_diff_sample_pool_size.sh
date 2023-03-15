#python train.py --seed 42 --simulation_env seals/MountainCar-v0 --po_algo ppo --sample_strategy cos_diff --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/HalfCheetah-v0 --po_algo ppo --sample_strategy cos_diff --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/MountainCar-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/HalfCheetah-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/MountainCar-v0 --po_algo ppo --sample_strategy random --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/HalfCheetah-v0 --po_algo ppo --sample_strategy random --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/Swimmer-v0 --po_algo ppo --sample_strategy cos_diff --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/Swimmer-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 1024
#
#python train.py --seed 42 --simulation_env seals/Swimmer-v0 --po_algo ppo --sample_strategy random --demo_batch_size 512 --sample_pool_size 1024

python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --sample_strategy cos_diff --demo_batch_size 512 --sample_pool_size 1024

python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 1024

python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --sample_strategy random --demo_batch_size 512 --sample_pool_size 1024
