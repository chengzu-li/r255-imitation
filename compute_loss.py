import pickle

file_path = r"outputs/seals/Ant-v0/ppo-mlp_policy/cos_diff-sample_512_from_2048-repeat3/loss.pickle"

with open(file_path, "rb") as f:
    loss = pickle.load(f)
print(len(loss))