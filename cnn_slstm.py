import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from google.colab import files
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


df_1 = pd.read_excel(r'/content/drive/MyDrive/energy_data/Bareily2019-21_(WeatherData).xlsx',engine='openpyxl')

print(df_1.info())

# This is the Usage file
df_2 = pd.read_excel(r'/content/drive/MyDrive/energy_data/BR_09.xlsx',engine='openpyxl')
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
class CNN_LSTM_NET(nn.Module):
    def __init__(self):
        super(CNN_LSTM_NET, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=4, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(128, 1)

    def forward(self, x, x1):
        # x is (batch_size, seq_length, feature_dim), swap axes for CNN
        # Reshape x to add a sequence dimension of size 1
        x = x.unsqueeze(1)  # Now x has shape (batch_size, 1, feature_dim)

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, feature_dim, seq_length)

        x = self.cnn(x)  # Apply CNN layers
        x = x.permute(0, 2, 1)  # Change shape back for LSTM (batch_size, seq_length, feature_dim)

        x, _ = self.lstm(x)  # LSTM processing
        x = self.fc(x[:, -1, :])  # Take last time step output

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = CNN_LSTM_NET().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
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

# List of years for which you have data
years = [2019, 2020, 2021]

# Create a single figure for all plots
fig, axs = plt.subplots(3 * len(years), 1, figsize=(12, 36), sharex=True)

# Loop through each year
for i, year in enumerate(years):
    # Load the data for 3-minute, 15-minute, and 60-minute intervals
    time_3 = np.load(f'time_day_3_{year}.npy')
    pred_3 = np.load(f'pred_day_3_{year}.npy')
    real_3 = np.load(f'real_day_3_{year}.npy')

    time_15 = np.load(f'time_day_15_{year}.npy')
    pred_15 = np.load(f'pred_day_15_{year}.npy')
    real_15 = np.load(f'real_day_15_{year}.npy')

    time_60 = np.load(f'time_day_60_{year}.npy')
    pred_60 = np.load(f'pred_day_60_{year}.npy')
    real_60 = np.load(f'real_day_60_{year}.npy')

    # Plot for 3-minute intervals
    axs[i * 3].plot(time_3, pred_3, label='Predicted', color='blue')
    axs[i * 3].plot(time_3, real_3, label='Actual', color='green')
    axs[i * 3].set_title(f'BR09 Day-Wise Electricity Consumption (3-Minute Intervals) - {year}')
    axs[i * 3].legend()

    # Plot for 15-minute intervals
    axs[i * 3 + 1].plot(time_15, pred_15, label='Predicted', color='blue')
    axs[i * 3 + 1].plot(time_15, real_15, label='Actual', color='green')
    axs[i * 3 + 1].set_title(f'BR09 Day-Wise Electricity Consumption (15-Minute Intervals) - {year}')
    axs[i * 3 + 1].legend()

    # Plot for 60-minute intervals
    axs[i * 3 + 2].plot(time_60, pred_60, label='Predicted', color='blue')
    axs[i * 3 + 2].plot(time_60, real_60, label='Actual', color='green')
    axs[i * 3 + 2].set_title(f'BR09 Day-Wise Electricity Consumption (60-Minute Intervals) - {year}')
    axs[i * 3 + 2].legend()

# Set common labels for the x-axis and title
fig.suptitle('Day-Wise Electricity Consumption for Multiple Years')
fig.text(0.5, 0.04, 'Time (minutes)', ha='center')
fig.text(0.04, 0.5, 'Electricity Consumption', va='center', rotation='vertical')

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the entire figure
plt.savefig('cnn+lstm__BR09_30E.png')

# Show the plots
plt.show()

# Load the first .npy file
data1 = np.load('real_3_2021.npy')
np.savetxt('real_3_2021.csv', data1, delimiter=',')
data2 = np.load('pred_3_2021.npy')
np.savetxt('pred_3_2021.csv', data2, delimiter=',')
data3 = np.load('real_15_2021.npy')
np.savetxt('real_15_2021.csv', data3, delimiter=',')
data4 = np.load('pred_15_2021.npy')
np.savetxt('pred_15_2021.csv', data4, delimiter=',')
data5 = np.load('real_60_2021.npy')
np.savetxt('real_60_2021.csv', data5, delimiter=',')
data6 = np.load('pred_60_2021.npy')
np.savetxt('pred_60_2021.csv', data6, delimiter=',')
data7 = np.load('real_day_60_2021.npy')
np.savetxt('real_day_60_2021.csv', data7, delimiter=',')
data8 = np.load('pred_day_60_2021.npy')
np.savetxt('pred_day_60_2021.csv', data8, delimiter=',')
data9 = np.load('real_day_15_2021.npy')
np.savetxt('real_day_15_2021.csv', data9, delimiter=',')
data10 = np.load('pred_day_15_2021.npy')
np.savetxt('pred_day_15_2021.csv', data10, delimiter=',')
data11 = np.load('real_day_3_2021.npy')
np.savetxt('real_day_3_2021.csv', data11, delimiter=',')
data12 = np.load('pred_day_3_2021.npy')
np.savetxt('pred_day_3_2021.csv', data12, delimiter=',')


data13 = np.load('real_3_2020.npy')
np.savetxt('real_3_2020.csv', data13, delimiter=',')
data14 = np.load('pred_3_2020.npy')
np.savetxt('pred_3_2020.csv', data14, delimiter=',')
data15 = np.load('real_15_2020.npy')
np.savetxt('real_15_2020.csv', data15, delimiter=',')
data16 = np.load('pred_15_2020.npy')
np.savetxt('pred_15_2020.csv', data16, delimiter=',')
data17 = np.load('real_60_2020.npy')
np.savetxt('real_60_2020.csv', data17, delimiter=',')
data18 = np.load('pred_60_2020.npy')
np.savetxt('pred_60_2020.csv', data18, delimiter=',')
data19 = np.load('real_day_60_2020.npy')
np.savetxt('real_day_60_2020.csv', data19, delimiter=',')
data20 = np.load('pred_day_60_2020.npy')
np.savetxt('pred_day_60_2020.csv', data20, delimiter=',')
data21 = np.load('real_day_15_2020.npy')
np.savetxt('real_day_15_2020.csv', data21, delimiter=',')
data22 = np.load('pred_day_15_2020.npy')
np.savetxt('pred_day_15_2020.csv', data22, delimiter=',')
data23 = np.load('real_day_3_2020.npy')
np.savetxt('real_day_3_2020.csv', data23, delimiter=',')
data24 = np.load('pred_day_3_2020.npy')
np.savetxt('pred_day_3_2020.csv', data24, delimiter=',')


data25 = np.load('real_3_2019.npy')
np.savetxt('real_3_2019.csv', data25, delimiter=',')
data26 = np.load('pred_3_2019.npy')
np.savetxt('pred_3_2019.csv', data26, delimiter=',')
data27 = np.load('real_15_2019.npy')
np.savetxt('real_15_2019.csv', data27, delimiter=',')
data28 = np.load('pred_15_2019.npy')
np.savetxt('pred_15_2019.csv', data28, delimiter=',')
data29 = np.load('real_60_2019.npy')
np.savetxt('real_60_2019.csv', data29, delimiter=',')
data30 = np.load('pred_60_2019.npy')
np.savetxt('pred_60_2019.csv', data30, delimiter=',')
data31 = np.load('real_day_15_2019.npy')
np.savetxt('real_day_15_2019.csv', data31, delimiter=',')
data32 = np.load('pred_day_15_2019.npy')
np.savetxt('pred_day_15_2019.csv', data32, delimiter=',')
data33 = np.load('real_day_60_2019.npy')
np.savetxt('real_day_60_2019.csv', data33, delimiter=',')
data34 = np.load('pred_day_60_2019.npy')
np.savetxt('pred_day_60_2019.csv', data34, delimiter=',')
data35 = np.load('real_day_3_2019.npy')
np.savetxt('real_day_3_2019.csv', data35, delimiter=',')
data36 = np.load('pred_day_3_2019.npy')
np.savetxt('pred_day_3_2019.csv', data36, delimiter=',')
# Create a DataFrame for better handling
data37 = np.load('Epoch_loss.npy')
np.savetxt('Epoch_loss.csv', data37, delimiter=',')


# Define file names
file_names = [


    'real_3_2020.csv', 'pred_3_2020.csv', 'real_15_2020.csv', 'pred_15_2020.csv',
    'real_60_2020.csv', 'pred_60_2020.csv', 'real_day_60_2020.csv', 'pred_day_60_2020.csv',
    'real_day_15_2020.csv', 'pred_day_15_2020.csv', 'real_day_3_2020.csv', 'pred_day_3_2020.csv',
    'real_3_2019.csv', 'pred_3_2019.csv', 'real_15_2019.csv', 'pred_15_2019.csv',
    'real_60_2019.csv', 'pred_60_2019.csv', 'real_day_15_2019.csv', 'pred_day_15_2019.csv',
    'real_day_60_2019.csv', 'pred_day_60_2019.csv', 'real_day_3_2019.csv', 'pred_day_3_2019.csv',
    'Epoch_loss.csv', 'real_3_2021.csv', 'pred_3_2021.csv', 'real_15_2021.csv', 'pred_15_2021.csv',
    'real_60_2021.csv', 'pred_60_2021.csv', 'real_day_60_2021.csv', 'pred_day_60_2021.csv',
    'real_day_15_2021.csv', 'pred_day_15_2021.csv', 'real_day_3_2021.csv', 'pred_day_3_2021.csv',
]


# Create an empty DataFrame to store the concatenated data
combined_data = pd.DataFrame()


# Load each CSV into a DataFrame and update column name
for file in file_names:
    df = pd.read_csv(file)
    column_name = file.split('.')[0]  # Extract column name from file name
    combined_data[column_name] = df.iloc[:, 0]  # Assuming all files have only one column


# Save the combined DataFrame to a single CSV file
combined_data.to_csv('cnn+lstm__BR09_30E.csv')
files.download('cnn+lstm__BR09_30E.csv')

# Load the combined data
combined_data = pd.read_csv('cnn+lstm__BR09_30E.csv', index_col=0)

# Extract the real and predicted values for 2021 and remove NaN values
real_3_2021 = combined_data['real_3_2021'].dropna().values
pred_3_2021 = combined_data['pred_3_2021'].dropna().values

real_15_2021 = combined_data['real_15_2021'].dropna().values
pred_15_2021 = combined_data['pred_15_2021'].dropna().values

real_60_2021 = combined_data['real_60_2021'].dropna().values
pred_60_2021 = combined_data['pred_60_2021'].dropna().values

# Calculate RMSE and MAE for 2021 data
rmse_3_2021 = np.sqrt(mean_squared_error(real_3_2021, pred_3_2021))
mae_3_2021 = mean_absolute_error(real_3_2021, pred_3_2021)

rmse_15_2021 = np.sqrt(mean_squared_error(real_15_2021, pred_15_2021))
mae_15_2021 = mean_absolute_error(real_15_2021, pred_15_2021)

rmse_60_2021 = np.sqrt(mean_squared_error(real_60_2021, pred_60_2021))
mae_60_2021 = mean_absolute_error(real_60_2021, pred_60_2021)

print(f"2021 3 min RMSE: {rmse_3_2021}, MAE: {mae_3_2021}")
print(f"2021 15 min RMSE: {rmse_15_2021}, MAE: {mae_15_2021}")
print(f"2021 60 min RMSE: {rmse_60_2021}, MAE: {mae_60_2021}")
