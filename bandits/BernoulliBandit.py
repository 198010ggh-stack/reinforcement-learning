#导入需要的库
import numpy as np
import matplotlib.pyplot as plt
#创建一个老虎机
class BernoulliBandit: #类
	"""伯努利多臂老虎机，输入K表示拉杆个数"""
	def __init__(self,K): #self是对象
		self.probs = np.random.uniform(size=K) #随机生成K个0-1的数，作为各跟拉杆的
		#中奖概率
		self.best_idx = np.argmax(self.probs) #返回概率最大值的索引
		self.best_prob = self.probs[self.best_idx] #找到最大的获奖概率
		self.K = K
	def step(self,k): #对象和选择的拉杆索引
		#拉一次指定的拉杆，返回这次的奖励：中奖是 1，没中奖是 0。
		if np.random.rand() < self.probs[k]:
			return 1
		else:
			return 0 #通过生成随机数的方法，来模拟80%的中奖概率

np.random.seed(1)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

#开始求解
class Solver: #求解器
	""" 多臂老虎机算法基本框架 """
	def __init__(self,bandit):
		self.bandit = bandit
		self.counts = np.zeros(self.bandit.K)#创建一个数组分别记录各拉杆次数
		self.regret = 0
		self.actions = []#创建个列表，放每次拉动竿的编号
		self.regrets = []#记录每步的累计误差

	def update_regret(self,k):
		self.regret += self.bandit.best_prob - self.bandit.probs[k]
		self.regrets.append(self.regret)

	def run_one_step(self):
		raise NotImplementedError

	def run(self,num_steps):
		for _ in range(num_steps):
			k = self.run_one_step()
			self.counts[k] += 1
			self.actions.append(k)
			self.update_regret(k)

# class EpsilonGreedy(Solver):
# 	"""epsilon算法"""
# 	def __init__(self, bandit, epsilon = 0.01,init_prob = 1.):
# 		super().__init__(bandit)
# 		self.epsilon = epsilon
# 		#所有拉杆的奖励估值
# 		self.estimates = np.array([init_prob]*self.bandit.K)

# 	def run_one_step(self):
# 		if np.random.random() < self.epsilon:
# 			k = np.random.randint(0,self.bandit.K)
# 		else:
# 			k = np.argmax(self.estimates)
# 		r = self.bandit.step(k)
# 		self.estimates[k] += 1./(self.counts[k]+1)*(r - self.estimates[k])
# 		return k

def plot_results(solvers,solver_name):
	"""生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。"""
	for idx,solver in enumerate(solvers):
		time_list = range(len(solver.regrets))
		plt.plot(time_list,solver.regrets,label = solver_name[idx])
	plt.xlabel('Time steps')
	plt.ylabel('Cumulate regrets')
	plt.title('%d-armed bandit' %solver.bandit.K)
	plt.legend() #显示图例
	plt.show() #显示整张图

# np.random.seed(1)
# epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm,epsilon = 0.01)
# epsilon_greedy_solver.run(5000)
# print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
# plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
		
# np.random.seed(0)
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# espilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm,epsilon=e)for e in epsilons]
# epsilons_greedy_names = ["epsilon={}".format(e) for e in epsilons]
# for solver in espilon_greedy_solver_list:
# 	solver.run(5000)

# plot_results(espilon_greedy_solver_list,epsilons_greedy_names)

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
