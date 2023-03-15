python train.py --seed 42 --simulation_env seals/MountainCar-v0 --po_algo trpo --sample_strategy cos_diff --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/HalfCheetah-v0 --po_algo trpo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env seals/Swimmer-v0 --po_algo trpo --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048
