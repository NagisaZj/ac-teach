import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas

def smooth(data, smooth_range):
	# print('hhhhhhh', type(data), len(data))
	new_data = np.zeros_like(data)
	for i in range(0, data.shape[-1]):
		if i < smooth_range:
			new_data[:, i] = 1. * np.sum(data[:, :i + 1], axis=1) / (i + 1)
		else:
			new_data[:, i] = 1. * np.sum(data[:, i - smooth_range + 1:i + 1], axis=1) / smooth_range

	return new_data

def read_csv(paths=[]):
    datas =[]
    for p in paths:
        with open(p,'r') as f:
            data=pandas.read_csv(f)
            print(data.keys())
            w=[]
            w.append(smooth(data['r'][None,:],100)[0])
            w.append(data['l'])
            w.append(data['t'])
            w.append(smooth(data['success'][None,:],100)[0])
            datas.append(w)
    return datas



# d1 = read_csv(['/data2/zj/ac-teach/logs/OneGoalPickPlaceDenseEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_0/gym_eval.monitor.csv'])
#d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_2/gym_eval.monitor.csv'])
# d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_6/monitor2.csv'])
# d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_7/monitor2.csv'])
#
# d1 = read_csv(['/data2/zj/ac-teach/logs/OneGoalPickPlaceDenseEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_0/gym_eval.monitor.csv'])
# d2 = read_csv(['/data2/zj/ac-teach/logs/OneGoalPickPlaceDenseEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_0/monitor.csv'])

d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_10/monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_10/monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_15/gym_eval.monitor2.csv'])

d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_18/gym_eval.monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_18/monitor2.csv'])

d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_20/gym_eval.monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_20/monitor2.csv'])

d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_22/gym_eval.monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_22/monitor2.csv'])

d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_20/gym_eval.monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_22/gym_eval.monitor2.csv'])
#
d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_24/gym_eval.monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_24/monitor2.csv'])
#
d1 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_31/gym_eval.monitor2.csv'])
d2 = read_csv(['/data2/zj/ac-teach/logs/MetaWorldEnv-v0/efficiency_partial_complete_suboptimal_ours/seed_31/monitor2.csv'])

plt.figure()
# plt.plot(range(len(d1[0][0])),d1[0][0])
# plt.plot(range(len(d2[0][0])),d2[0][0],color='r')
plt.plot(np.arange(d1[0][0].shape[0])*500,d1[0][0],label='Push, Push Source, Test')
plt.plot(np.arange(d2[0][0].shape[0])*500,d2[0][0],color='r',label='Push, Push Source, Train')
plt.legend()
plt.show()
