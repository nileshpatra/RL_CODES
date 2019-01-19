from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 

class Bandit:
	def __init__(self , m):
		self.m = m #true mean
		self.mean = 0
		self.N = 0

	def pull(self):
		return np.random.randn() + self.m

	def push(self , x):
		self.N += 1
		self.mean = (1 - (1.0)/self.N) * self.mean + 1.0/self.N*x


def experiment(m1 , m2 , m3 , eps , N):
	b1 , b2 , b3 = Bandit(m1) , Bandit(m2) , Bandit(m3)
	bandits = [b1 , b2 , b3]
	data = []
	for _ in range(N):
		p = np.random.random()
		if p < eps:
			pos = np.random.randint(0 , 3)
		else:
			pos = np.argmax([b.mean for b in bandits])

		retval = bandits[pos].pull()
		bandits[pos].push(retval)
		data.append(retval)
	data = np.array(data)
	cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

	for b in bandits:
			print(b.mean)

	return cumulative_average

c_1 = experiment(1.0, 2.0, 3.0, 0.1, 1000000)
c_05 = experiment(1.0, 2.0, 3.0, 0.05, 1000000)
c_2 = experiment(1.0, 2.0, 3.0, 0.2, 1000000)


plt.plot(c_1, label='eps = 0.1')
plt.plot(c_05, label='eps = 0.05')
plt.plot(c_2, label='eps = 0.2')
plt.legend()
plt.xscale('log')
plt.show()




