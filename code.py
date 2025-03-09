import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from google.colab import files

# This is the Weather File
df_1 = pd.read_excel(r'/content/drive/MyDrive/energy_data/Bareily2019-21_(WeatherData).xlsx',engine='openpyxl')

print(df_1.info())

# This is the Usage file
df_2 = pd.read_excel(r'/content/drive/MyDrive/energy_data/BR_06.xlsx',engine='openpyxl')
print(df_2.info())


# Drop the not active cells or missing values cells
# df_2 = df_2.dropna()
# df_1 = df_1.dropna()

# This should be done in year wise then
def collect_data_year_wise(year_number):

	Date_name = 'Date_'+str(year_number)
	Time_name = 'Time_'+str(year_number)

	new_date = []
	year = []
	month = []
	date = []
	new_time = []
	A = []
	B = []
	C = []
	D = []
	see = df_2[Date_name]

	see = see.dropna()

	# separate the dates as they are not is same format in Usage and weather files
	for i in see:
		i = str(i)
		c = i.split(' ')
		new_date.append(c[0])
		h = c[0].split('-')
		year.append(h[0])
		month.append(h[1])
		date.append(h[2])


	# This is sorting time
	see = df_2[Time_name]
	see = see.dropna()

	for i in see:
		i = str(i)
		new_time.append(i)

	# electricity name
	elec_name = 'kWh_'+str(year_number)

	electricity_use = []
	see = df_2[elec_name]
	see = see.dropna()

	for i in see:
		# i = str(i)
		electricity_use.append(float(i))


	Avg_volt = []
	avgv_name = 'Avg Voltage_'+str(year_number)

	see = df_2[avgv_name]
	see = see.dropna()

	for i in see:
		# i = str(i)
		Avg_volt.append(float(i))


	Avg_curr = []
	avgc_name = 'Avg Current_'+str(year_number)
	see = df_2[avgc_name]
	see = see.dropna()

	for i in see:
		# i = str(i)
		Avg_curr.append(float(i))


	freq = []
	freq_name = 'Freq_'+str(year_number)
	see = df_2[freq_name]
	see = see.dropna()

	for i in see:
		# i = str(i)
		freq.append(float(i))

	# Separated dates and there usages
	return year,month,date,new_time,electricity_use,Avg_volt,Avg_curr,freq

# Must also load the Time and Weather values
def get_loaded_list_values(year_number):

	year,month,date,new_time,electricity_use,Avg_volt,Avg_curr,freq = collect_data_year_wise(year_number)

	print(len(year),year_number)

	# Input data points
	nc = []
	nc_1 = []
	# Output Prediction
	elect = []

	for count in range(len(year)):

		a = int(year[count])
		b = int(month[count])
		c = int(date[count])
		d = new_time[count]
		h = int(d.split(":")[0])
		m = int(d.split(":")[1])
		e = electricity_use[count]


		get_date = str(c)+'-'+str(b)+'-'+str(a)
		new_date = 'Date_'+str(year_number)
		new_hour = 'HR_'+str(year_number)
		curr_val = (df_1.loc[df_1[new_date] == get_date])
		c_t = (curr_val.loc[curr_val[new_hour] == (h)])
		d = c_t.index.tolist()[0]


		temp_name = ' Temperature at 2 Meters (C)_'+str(year_number)
		wet_name = 'Wet Bulb Temperature at 2 Meters (C)_'+str(year_number)
		perc_name = 'Precipitation Corrected (mm/hour)_'+str(year_number)
		hum_name = 'Relative Humidity at 2 Meters (%)_'+str(year_number)
		wind_name = 'Wind Speed at 10 Meters (m/s)_'+str(year_number)
		wind_name_dir = 'Wind Direction at 10 Meters (Degrees)_'+str(year_number)

		temperature = []

		temp = c_t[temp_name]
		wet = c_t[wet_name]
		perc = c_t[perc_name]
		hum = c_t[hum_name]
		wind = c_t[wind_name]
		wind_dir = c_t[wind_name_dir]

		nc.append([temp[d],wet[d],perc[d],hum[d],wind[d],wind_dir[d],Avg_volt[count],Avg_curr[count],freq[count]])
		nc_1.append([a-2019,b,c,h,m])
		elect.append(e)

	nc = np.array(nc)
	nc_1 = np.array(nc_1)
	elect = np.array(elect)
	elect = np.reshape(elect,(elect.shape[0],1))


	return nc,nc_1,elect

nc_2019,nc_1_2019,elect_2019 = get_loaded_list_values(2019)
print(nc_2019.shape,nc_1_2019.shape,elect_2019.shape)

nc_2020,nc_1_2020,elect_2020 = get_loaded_list_values(2020)
print(nc_2020.shape,nc_1_2020.shape,elect_2020.shape)

nc_2021,nc_1_2021,elect_2021 = get_loaded_list_values(2021)
print(nc_2021.shape,nc_1_2021.shape,elect_2021.shape)

nc = [nc_2019,nc_2020,nc_2021]
nc_1 = [nc_1_2019,nc_1_2020,nc_1_2021]
elect = [elect_2019,elect_2020,elect_2021]

# nc = [nc_2019,nc_2019,nc_2019]
# nc_1 = [nc_1_2019,nc_1_2019,nc_1_2019]
# elect = [elect_2019,elect_2019,elect_2019]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class emb(nn.Module):

	def __init__(self):
		super().__init__()

		self.encoder = torch.nn.Embedding(100,128)

	def forward(self,x):

		embs = (self.encoder(x))
		embs = embs.squeeze(dim = 1)

		return embs


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        self.year = emb()
        self.date = emb()
        self.month = emb()
        self.hour = emb()
        self.minute = emb()

        self.seq = rnn = nn.LSTM(9 + 128 * 5, 256, 3)
        self.dropout = nn.Dropout(0.4)  # You can adjust the dropout rate (0.4 as an example)
        self.l1 = nn.Linear(256, 1)


		# self.seq = nn.Sequential(nn.Linear(5+128,128),nn.ReLU(),
		# 						nn.Linear(128,128),nn.ReLU(),
		# 						nn.Linear(128,128),nn.ReLU(),
		# 						nn.Linear(128,1))
    def forward(self, x, x1):
        y = x1[:, 0]
        d = x1[:, 1]
        m = x1[:, 2]
        h = x1[:, 3]
        mi = x1[:, 4]

        y = y.reshape(-1, 1)
        d = d.reshape(-1, 1)
        m = m.reshape(-1, 1)
        h = h.reshape(-1, 1)
        mi = mi.reshape(-1, 1)

        y = self.year(y)
        d = self.date(d)
        m = self.date(m)
        h = self.date(h)
        mi = self.date(mi)

        y = self.dropout(y)
        d = self.dropout(d)
        m = self.dropout(m)
        h = self.dropout(h)
        mi = self.dropout(mi)

        x = torch.cat((x, y, d, m, h, mi), dim=1)
        x = x.reshape([-1, 5, 128 * 5 + 9])
        x, (hidden, cell_state) = self.seq(x)
        x = self.l1(x)
        x = x.reshape(-1, 1)

        return x


device = 'cuda:0'

net = NET().to(device)

optimizer = optim.Adam(net.parameters(),lr = 0.0001)
criterion_l2 = nn.MSELoss()

iter_counts = []
rmse_val = []
pred_val_t = []
pred_val_p = []
final_count = 0
epoch_loss = []

curr_count = 0

for epochs in range(30):
	print("EPOCH:", epochs+1)

	# This is the epoch loss
	check_loss = 0

	for years in range(2):

		nc_c = nc[years]
		nc_1_c = nc_1[years]
		elect_c = elect[years]

		in_count = 0
		for count in range(int(nc_c.shape[0]/100)-5):

			# if(count == 100):
			# 	break

			x = nc_c[in_count:in_count+100]
			x1 = nc_1_c[in_count:in_count+100]
			y = elect_c[in_count:in_count+100]

			# print(x1)

			in_count = in_count + 100
			curr_count = curr_count + 1

			x = torch.tensor(x).float().to(device)
			x1 = torch.tensor(x1).long().to(device)
			y = torch.tensor(y).float().to(device)

			optimizer.zero_grad()

			if(x.shape[0] < 100):
				continue

			val = net(x,x1)

			loss = criterion_l2(val,y)
			loss.backward()

			optimizer.step()

			with torch.no_grad():
				check_loss += loss

			with torch.no_grad():

				if(count % 100 == 0):
					print('========================================================================================')
					print(check_loss,count)
					print('error in predictions')
					for l in range(5):
						print('This is the time---->',x1[l])
						print('pred-->',val[l].item(),',gt--->',y[l].item(),',diff--->',val[l].item()-y[l].item())

				iter_counts.append(curr_count)
				rmse_val.append(loss.cpu().numpy())
				pred_val_t.append(val[l].cpu().numpy())
				pred_val_p.append(y[l].cpu().numpy())

	with torch.no_grad():

		epoch_loss.append(check_loss.cpu().numpy())

		state = {'net_0' : net.state_dict(),'net_0_opt' : optimizer.state_dict()}
		torch.save(state,'./state.pt')



# Values to store in numpy values
np.save('./Epoch_loss.npy',epoch_loss)
np.save('./iter_count_3.npy',iter_counts)
np.save('./rmse_val_3_minutes.npy',rmse_val)
np.save('./pred_val_3_minutes.npy',pred_val_p)
np.save('./real_val_3_minutes.npy',pred_val_t)


# Number = (mintues / 3)
def get_values(number):

	i1 = []
	r1 = []
	p1 = []
	p2 = []

	for i in range(len(iter_counts)-number+2):

		if (i % number == 0):
			i1.append(iter_counts[i])

			a = 0
			for j in range(number):
				a = pred_val_p[i+j].item() + a
				p1.append(a)

			a = 0
			for j in range(number):
				a = pred_val_t[i+j].item() + a
				p2.append(a)

			a = 0
			for j in range(number):
				a = rmse_val[i+j].item() + a
				r1.append(a)


	return i1,r1,p1,p2


i1,r1,p1,p2 = get_values(5)
# Values to store in numpy values
np.save('./iter_count_15.npy',i1)
np.save('./rmse_val_15_minutes.npy',r1)
np.save('./pred_val_15_minutes.npy',p1)
np.save('./real_val_15_minutes.npy',p2)

i1,r1,p1,p2 = get_values(20)
# Values to store in numpy values
np.save('./iter_count_60.npy',i1)
np.save('./rmse_val_60_minutes.npy',r1)
np.save('./pred_val_60_minutes.npy',p1)
np.save('./real_val_60_minutes.npy',p2)


# Need to plot year
# Data is already sorted based on the timestamp and dates given in the data
# Loading of data in pandas is sequential
def calculate_year_wise_plots(year):

	iter_counts_y = []
	rmse_val_y = []
	pred_val_y = []
	real_val_y = []
	final_count = 0
	time = 0

	# This is for indexing in the list [0,1,2] ---> [2019,2020,2021]
	nc_c = nc[year-2019]
	nc_1_c = nc_1[year-2019]
	elect_c = elect[year-2019]

	in_count = 0

	print(nc_c.shape,'This is the number of time stamps for the year ',year)

	# Stop Backpropagation
	with torch.no_grad():

		for count in range(int(nc_c.shape[0]/100)-5):

			x = nc_c[in_count:in_count+100]
			x1 = nc_1_c[in_count:in_count+100]
			y = elect_c[in_count:in_count+100]

			# print(x1)

			in_count = in_count + 100

			x = torch.tensor(x).float().to(device)
			x1 = torch.tensor(x1).long().to(device)
			y = torch.tensor(y).float().to(device)

			if(x.shape[0] < 100):
				continue

			val = net(x,x1)

			loss = criterion_l2(val,y)

			for l in range(100):

				# Adding time... as all the values predicted are by 3 mintues GAP
				iter_counts_y.append(time)
				time = time + 3
				pred_val_y.append(val[l].cpu().numpy())
				real_val_y.append(y[l].cpu().numpy())

	mins_name = 'time_3_'+str(year)+'.npy'
	pred_name = 'pred_3_'+str(year)+'.npy'
	real_name = 'real_3_'+str(year)+'.npy'

	np.save(mins_name,iter_counts_y)
	np.save(pred_name,pred_val_y)
	np.save(real_name,real_val_y)

	t1 = []
	p1 = []
	r1 = []

	p_c = 0
	r_c = 0
	for i in range(int(len(iter_counts_y))-1):

		p_c = p_c + pred_val_y[i]
		r_c = r_c + real_val_y[i]

		if(i % 5 == 0):
			t1.append(iter_counts_y[i])
			p1.append(p_c)
			r1.append(r_c)
			p_c = 0
			r_c = 0

	mins_name = 'time_15_'+str(year)+'.npy'
	pred_name = 'pred_15_'+str(year)+'.npy'
	real_name = 'real_15_'+str(year)+'.npy'

	np.save(mins_name,t1)
	np.save(pred_name,p1)
	np.save(real_name,r1)


	p1 = []
	r1 = []
	t1 = []

	p_c = 0
	r_c = 0
	for i in range(int(len(iter_counts_y))-1):

		p_c = p_c + pred_val_y[i]
		r_c = r_c + real_val_y[i]

		if(i % 20 == 0):
			t1.append(iter_counts_y[i])
			p1.append(p_c)
			r1.append(r_c)
			p_c = 0
			r_c = 0

	mins_name = 'time_60_'+str(year)+'.npy'
	pred_name = 'pred_60_'+str(year)+'.npy'
	real_name = 'real_60_'+str(year)+'.npy'

	np.save(mins_name,t1)
	np.save(pred_name,p1)
	np.save(real_name,r1)

	return


calculate_year_wise_plots(2019)
calculate_year_wise_plots(2020)
calculate_year_wise_plots(2021)


def calculate_day_wise_plots(year):

	iter_counts_y = []
	rmse_val_y = []
	pred_val_y = []
	real_val_y = []
	final_count = 0
	time = 0

	# This is for indexing in the list [0,1,2] ---> [2019,2020,2021]
	nc_c = nc[year-2019]
	nc_1_c = nc_1[year-2019]
	elect_c = elect[year-2019]

	in_count = 0

	print(nc_c.shape,'This is the number of time stamps for the year ',year)

	# Stop Backpropagation
	with torch.no_grad():

		for count in range(int(nc_c.shape[0]/100)-5):

			if(count == 5):
				break

			x = nc_c[in_count:in_count+100]
			x1 = nc_1_c[in_count:in_count+100]
			y = elect_c[in_count:in_count+100]

			# print(x1)

			in_count = in_count + 100

			x = torch.tensor(x).float().to(device)
			x1 = torch.tensor(x1).long().to(device)
			y = torch.tensor(y).float().to(device)

			if(x.shape[0] < 100):
				continue

			val = net(x,x1)

			loss = criterion_l2(val,y)

			for l in range(100):

				# Adding time... as all the values predicted are by 3 mintues GAP
				iter_counts_y.append(time)
				time = time + 3
				pred_val_y.append(val[l].cpu().numpy())
				real_val_y.append(y[l].cpu().numpy())

	iter_counts_y = iter_counts_y[:480]
	pred_val_y = pred_val_y[:480]
	real_val_y = real_val_y[:480]

	mins_name = 'time_day_3_'+str(year)+'.npy'
	pred_name = 'pred_day_3_'+str(year)+'.npy'
	real_name = 'real_day_3_'+str(year)+'.npy'

	np.save(mins_name,iter_counts_y)
	np.save(pred_name,pred_val_y)
	np.save(real_name,real_val_y)

	t1 = []
	p1 = []
	r1 = []

	p_c = 0
	r_c = 0
	for i in range(int(len(iter_counts_y))):

		p_c = p_c + pred_val_y[i]
		r_c = r_c + real_val_y[i]

		if(i % 5 == 0):
			t1.append(iter_counts_y[i])
			p1.append(p_c)
			r1.append(r_c)
			p_c = 0
			r_c = 0

	mins_name = 'time_day_15_'+str(year)+'.npy'
	pred_name = 'pred_day_15_'+str(year)+'.npy'
	real_name = 'real_day_15_'+str(year)+'.npy'

	np.save(mins_name,t1)
	np.save(pred_name,p1)
	np.save(real_name,r1)


	p1 = []
	r1 = []
	t1 = []

	p_c = 0
	r_c = 0
	for i in range(int(len(iter_counts_y))):

		p_c = p_c + pred_val_y[i]
		r_c = r_c + real_val_y[i]

		if(i % 20 == 0):
			t1.append(iter_counts_y[i])
			p1.append(p_c)
			r1.append(r_c)
			p_c = 0
			r_c = 0

	mins_name = 'time_day_60_'+str(year)+'.npy'
	pred_name = 'pred_day_60_'+str(year)+'.npy'
	real_name = 'real_day_60_'+str(year)+'.npy'

	np.save(mins_name,t1)
	np.save(pred_name,p1)
	np.save(real_name,r1)

	return


calculate_day_wise_plots(2019)
calculate_day_wise_plots(2020)
calculate_day_wise_plots(2021)
