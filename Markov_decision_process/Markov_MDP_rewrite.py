import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]  # 狀態集合，s5 是終點
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

#定義策略
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
Pi_2 = {
	("s1", "保持s1"): 0.6,
	("s1", "前往s2"): 0.4,
	("s2", "前往s1"): 0.3,
	("s2", "前往s3"): 0.7,
	("s3", "前往s4"): 0.5,
	("s3", "前往s5"): 0.5,
	("s4", "前往s5"): 0.1,
	("s4", "機率前往"): 0.9,
}

Pi_3 = {
	("s1", "保持s1"): 0.,
	("s1", "前往s2"): 1.,
	("s2", "前往s1"): 1.,
	("s2", "前往s3"): 0.,
	("s3", "前往s4"): 1.,
	("s3", "前往s5"): 0.,
	("s4", "前往s5"): 1.,
	("s4", "機率前往"): 0.,
}

idx = {s:i for i,s in enumerate(S)}

#MDP->MRP
def mdp_mrp(P,R,Pi):
	n = len(S)
	R_mrp = np.zeros(n)
	P_mrp = np.zeros((n,n))
	for (s,a),pi in Pi.items():
		R_mrp[idx [s]] += pi*R[(s,a)]
	for(s,a,s2),p in P.items():
		P_mrp[idx[s],idx[s2]]+= Pi[(s,a)]*p
	return R_mrp,P_mrp
#求V
def compute_V(R_mrp,P_mrp,gamma):
	V = np.zeros(len(S))
	for i in range(1,1001):
		V = R_mrp + gamma*(P_mrp@V)
	return V
#求Q
def compute_Q(P,R,V,gamma):
	Q = {}
	for (s,a),r in R.items():
		future = 0
		for(s_,a_,s2),p in P.items():
			if s_ == s and a_==a:
				future+=p*V[idx[s2]]
		Q[(s,a)] = r + gamma*future
	return Q
#策略改進
def improve_policy(Q):
	best_a = {}
	for(s,a),q in Q.items():
		if s not in best_a or q >Q[(s,best_a[s])]:
			best_a[s] = a
	new_Pi = {}
	for(s,a) in Q:
		if a == best_a[s]:
			new_Pi[(s,a)] = 1.0
		else:
			new_Pi[(s,a)] = 0.
	return new_Pi
#策略迭代
def policy_iteration(P,R,Pi,gamma):
	for k in range(1,101):
		R_map,P_map = mdp_mrp(P,R,Pi)
		V = compute_V(R_map,P_map,gamma)
		Q = compute_Q(P,R,V,gamma)
		new_Pi = improve_policy(Q)
		print(f"第{k}輪：V={np.round(V,2)}")
		if new_Pi == Pi:
			break
		Pi = new_Pi
	return Pi,V



# R_map,P_map = mdp_mrp(P,R,Pi)
# V = compute_V(R_map,P_map,gamma)
# Q = compute_Q(P,R,V,gamma)
# print(np.round(V,2))
# for key,q in Q.items():
# 	print(f"{key}的動作價值是{round(q,2)}")
best_Pi,best_V = policy_iteration(P,R,Pi,gamma)
print("最優策略：")
for key,q in best_Pi.items():
	if q == 1.:
		print(f"{key}")


