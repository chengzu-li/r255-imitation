#python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --policy mlp --sample_strategy cos_diff --demo_batch_size 512 --sample_pool_size 2048
#
#python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --policy mlp --sample_strategy cos_sim --demo_batch_size 512 --sample_pool_size 2048
#
#python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --policy mlp --sample_strategy scheduling --demo_batch_size 512 --sample_pool_size 2048

python train.py --seed 42 --simulation_env Reacher-v2 --po_algo ppo --policy mlp --sample_strategy kmeans --demo_batch_size 512 --sample_pool_size 2048
