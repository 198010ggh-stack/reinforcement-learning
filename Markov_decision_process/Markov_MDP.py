import numpy as np

# ===== 第 1 步：定義 MDP =====
S = ["s1", "s2", "s3", "s4", "s5"]  # 狀態集合，s5 是終點
# 轉移機率 P(s'|s,a)，key = (狀態, 動作, 下一個狀態)
P = {
	("s1", "保持s1", "s1"): 1.0,
	("s1", "前往s2", "s2"): 1.0,
	("s2", "前往s1", "s1"): 1.0,
	("s2", "前往s3", "s3"): 1.0,
	("s3", "前往s4", "s4"): 1.0,
	("s3", "前往s5", "s5"): 1.0,
	("s4", "前往s5", "s5"): 1.0,
	("s4", "機率前往", "s2"): 0.2,
	("s4", "機率前往", "s3"): 0.4,
	("s4", "機率前往", "s4"): 0.4,
}
# 獎勵 r(s,a)，key = (狀態, 動作)
R = {
	("s1", "保持s1"): -1,
	("s1", "前往s2"): 0,
	("s2", "前往s1"): -1,
	("s2", "前往s3"): -2,
	("s3", "前往s4"): -2,
	("s3", "前往s5"): 0,
	("s4", "前往s5"): 10,
	("s4", "機率前往"): 1,
}
gamma = 0.5

# ===== 第 2 步：定義策略 π(a|s) =====
# 隨機策略：每個狀態的兩個動作各 50%
Pi = {
	("s1", "保持s1"): 0.5,
	("s1", "前往s2"): 0.5,
	("s2", "前往s1"): 0.5,
	("s2", "前往s3"): 0.5,
	("s3", "前往s4"): 0.5,
	("s3", "前往s5"): 0.5,
	("s4", "前往s5"): 0.5,
	("s4", "機率前往"): 0.5,
}

# 狀態名稱 → 陣列索引，例如 "s1" → 0、"s4" → 3
idx = {s: i for i, s in enumerate(S)}

# ===== 第 3 步：MDP → MRP =====
def mdp_to_mrp(P, R, Pi):
	"""把動作依策略平均掉，得到 MRP 的 r'(s) 和 P'(s'|s)"""
	n = len(S)
	R_mrp = np.zeros(n)
	P_mrp = np.zeros((n, n))
	# 公式①：r'(s) = Σ_a π(a|s)·r(s,a)
	for (s, a), pi in Pi.items():
		R_mrp[idx[s]] += pi * R[(s, a)]
	# 公式②：P'(s'|s) = Σ_a π(a|s)·P(s'|s,a)
	for (s, a, s2), p in P.items():
		P_mrp[idx[s], idx[s2]] += Pi[(s, a)] * p
	return R_mrp, P_mrp

# ===== 第 4 步：求 V（跟上一課一樣的迭代）=====
def compute_V(R_mrp, P_mrp, gamma):
	"""V(s) = r'(s) + γ·Σ_s' P'(s'|s)·V(s')，反覆迭代直到收斂"""
	V = np.zeros(len(R_mrp))
	for i in range(1000):
		V = R_mrp + gamma * (P_mrp @ V)
	return V

# ===== 第 5 步：由 V 求 Q（動作價值） =====
def compute_Q(P, R, V, gamma):
	"""Q(s,a) = r(s,a) + γ·Σ_s' P(s'|s,a)·V(s')"""
	Q = {}
	for (s, a), r in R.items():
		future = 0  # Σ_s' P(s'|s,a)·V(s')
		for (s_, a_, s2), p in P.items():
			if s_ == s and a_ == a:  # 只加總「這個 (s,a)」的轉移
				future += p * V[idx[s2]]
		Q[(s, a)] = r + gamma * future
	return Q
