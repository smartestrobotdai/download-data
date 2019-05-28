import sys
import os.path
sys.path.append("../")
from optimizeresult import OptimizeResult
from util import *


if len(sys.argv) < 5:
	print("usage: python3 search-worms.py stock_name, stock_index traning_start_day_index, training_end_day_index")
	sys.exit()

stock_name = sys.argv[1]
stock_index = int(sys.argv[2])
training_start_day_index = int(sys.argv[3])
training_end_day_index = int(sys.argv[4])

swarm_dir = get_swarm_dir(stock_name, stock_index, training_start_day_index, training_end_day_index)

strategy_file = os.path.join(swarm_dir, 'strategy_cache.txt')
print("Top 10 Strategies for {}: swarm: {}-{}".format(stock_name, training_start_day_index, training_end_day_index))
result_strategies = OptimizeResult(-1)
result_strategies.load(strategy_file)
result_strategies.get_best_results(10)

strategy_file = os.path.join(swarm_dir, 'stockworm_cache.txt')
print("Top 10 Worms for {}: swarm: {}-{}".format(stock_name, training_start_day_index, training_end_day_index))
optimize_results = OptimizeResult(-2)
optimize_results.load(strategy_file)
optimize_results.get_best_results(10)