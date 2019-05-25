import numpy as np
import os.path
import pandas as pd

class OptimizeResult:
	def __init__(self):
		self.data = None
		self.filename = "test.txt"
		return

	def insert_result(self, X, Y):
		new_data = [np.hstack((X,Y))]
		if self.data is None:
			self.data = np.array(new_data)
		else:
			cached_y, index = self.find_result(X)
			if cached_y == None:
				self.data = np.insert(self.data, index, new_data, axis=0)



	# return: the value and the position it should be inserted.
	def find_result(self, X):
		if self.data is None:
			return None, 0

		n_column = len(X)
		start = 0
		end = len(self.data) - 1

		while end >= start:
			#print("end:{}, start:{}".format(end, start))
			mid = (end + start) // 2
			comp_result = self.compare(self.data[mid, :n_column], X)

			if comp_result > 0:
				end = mid - 1
			elif comp_result < 0:
				start = mid + 1
			else:
				return self.data[mid, n_column:], mid

		return None, start

	def compare(self, x1, x2):
		assert(len(x1)==len(x2))

		if np.all(np.isclose(x1, x2)):
			return 0

		for i in range(len(x1)):
			if x1[i] < x2[i]:
				return -1
			elif x1[i] > x2[i]:
				return 1
		return 0

	def get_size(self):
		if self.data is None:
			return 0

		return len(self.data)

	def find_best_results(self, n_top_rows, by):
		if by < 0:
			n_columns = self.data.shape[1]
			by = n_columns + by

		df = pd.DataFrame(self.data)
		df = df.sort_values(by=by)
		print(df.tail(n_top_rows))
		return df.tail(n_top_rows).values

	def save(self, filename):
		np.savetxt(filename, self.data, delimiter=',')

	def load(self, filename):
		if not os.path.isfile(filename):
			print("cannot find file:{}".format(filename))
			return
		self.data = np.loadtxt(filename, delimiter=',')
		# only one line
		shape = self.data.shape
		if len(self.data.shape) == 1:
			self.data = self.data.reshape(1, shape[0])

if __name__ == '__main__':
	optimize_results = OptimizeResult()
	optimize_results.insert_result([1.0,2,3], [1,2])
	optimize_results.insert_result([2.0,3,4], [3,3])
	optimize_results.insert_result([4.0,3,5], [3,4])
	y, index = optimize_results.find_result([4.0000001,3,5])
	print(y, index)
	print("find strategies")
	result_strategies = OptimizeResult()
	result_strategies.load('strategy_cache.txt')
	result_strategies.find_best_results(10, by=-1)


	print("find worms")
	optimize_results = OptimizeResult()
	optimize_results.load('worm_cache.txt')
	optimize_results.find_best_results(100, by=-2)



