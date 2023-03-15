python train.py --seed 42 --simulation_env seals/CartPole-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/MountainCar-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/Ant-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/HalfCheetah-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

#python train.py --seed 42 --simulation_env seals/Hopper-v0 --po_algo ppo --sample_strategy random --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/Humanoid-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/Walker2d-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/Swimmer-v0 --po_algo ppo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048
