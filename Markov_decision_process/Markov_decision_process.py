import numpy as np
rewards = [-1,-2,-2,10,1,0]#狀態s1-s6各自的獎勵
# gamma = 0.9
# def compute_return(chain,gamma):
# 	"""計算一條狀態序列的回報"""
# 	G = 0
# 	for k in range(len(chain)):
# 		s = chain[k]
# 		r = rewards[s-1]
# 		G += (gamma**k)*r
# 	return G
# chain = [1,2,3,6] #狀態鏈路
# print(f"回報G ={compute_return(chain,gamma)}")

# 每一列 = 從某個狀態出發，走到各狀態的機率（每列加起來 = 1）
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  # 從 s1 出發
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],  # 從 s2 出發
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],  # 從 s3 出發
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],  # 從 s4 出發
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],  # 從 s5 出發
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # s6 是終點，永遠留在原地
])
R = np.array(rewards)
gamma = 0.5

V = np.zeros(len(rewards))
for i in range(1000):
	V = R + gamma*(P@V)
print(np.round(V,2))
