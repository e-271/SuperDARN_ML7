import sys
import ipdb
import json
import sqlite3
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.dates import DateFormatter
from matplotlib.dates import date2num, num2date

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from davitpy import utils


def gmm_dist(rad, stm, etm):
    
    #Read data with emprical model information and RTI plot###########################################################################################
	data_dict = read_from_updated_db(rad, stm, etm)

	plot_rti_emp(rad, stm, etm, data_dict,'velocity',fig_num=1,title_str='Empirical Model Results')

	gs_hops = [1.0, 2.0, 3.0]
	is_hops = [0.5, 1.5, 2.5]

	#emp_gsflg = np.hstack(data_dict['gsflg'])
	emp_gate = np.hstack(data_dict['gate'])
	emp_time, emp_gsflg = [], []
	emp_num_scatter = data_dict['num_scatter']

	for i in range(len(emp_num_scatter)):
		emp_time.extend(date2num([data_dict['datetime'][i]]*emp_num_scatter[i]))
		for j in range(len(data_dict['hop'][i])):
			if data_dict['hop'][i][j] in is_hops:
				emp_gsflg.append(0)
			elif data_dict['hop'][i][j] in gs_hops:
				emp_gsflg.append(1) 
 
	emp_gsflg = np.array(emp_gsflg)
	emp_time = np.array(emp_time)
    #Read data with emprical model information and RTI plot##########################################################################################

    #Read data from database and RTI plot###########################################################################################################
	plot_rti(rad, stm, etm, data_dict,'velocity',gsct=False,fig_num=2,title_str='Traditional Model Results')

	gate = np.hstack(data_dict['gate'])
	vel = np.hstack(data_dict['velocity'])
	wid = np.hstack(data_dict['width'])
	power = np.hstack(data_dict['power'])
	elev = np.hstack(data_dict['elevation'])
	gs_flg = np.hstack(data_dict['gsflg'])
    
  
   	# Extend time, freq, and beam so that all dimensions match 
	time, freq, beam = [], [], []
	num_scatter = data_dict['num_scatter']
	for i in range(len(num_scatter)):
		#date_time.extend([data_dict['datetime'][i]]*num_scatter[i])
		time.extend(date2num([data_dict['datetime'][i]]*num_scatter[i]))
		freq.extend([data_dict['frequency'][i]]*num_scatter[i])
		beam.extend([float(data_dict['beam'][i])]*num_scatter[i])
	time = np.array(time)
	freq = np.array(freq)
	beam = np.array(beam)
    #Read data from database and RTI plot#############################################################################################################



    #GMM and RTI plot#################################################################################################################################
    #need to scale data before apply kmeans

	beam_scaled = preprocessing.scale(beam)
	gate_scaled = preprocessing.scale(gate)
	vel_scaled = preprocessing.scale(vel)
	wid_scaled = preprocessing.scale(wid)
	power_scaled = preprocessing.scale(power)
	time_scaled = preprocessing.scale(time)
	elev_scaled = preprocessing.scale(elev)
	freq_scaled = preprocessing.scale(freq)
	

	data = np.column_stack((beam_scaled,gate_scaled,vel_scaled,wid_scaled,\
                                 power_scaled,elev_scaled,freq_scaled,time_scaled))
	N,D = data.shape
	n_classes = 3

	kmeans = KMeans(init = 'k-means++', n_clusters = n_classes, n_init=50).fit(data)    

    # source
    # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
	cov_type = 'full' # ['spherical', 'diag', 'tied', 'full']
	estimator = GaussianMixture(n_components=n_classes, \
                           covariance_type=cov_type, max_iter=100, \
                           random_state=0)
	# initialize the GMM parameters with kmean centroid
	estimator.means_init = kmeans.cluster_centers_ #np.random.random((n_classes, D))*2.0-1.0
    # Train the other parameters using the EM algorithm.
	estimator.fit(data)
	Z = estimator.predict(data)

	median_vels = np.zeros(n_classes)
	median_wids = np.zeros(n_classes)
	for i in range(n_classes):
		median_vels[i] = np.median(np.abs(vel[Z == i]))
		median_wids[i] = np.median(wid[Z == i])
		print median_vels[i]
		print median_wids[i]

	gsfg_min_vel = np.argmin(median_vels)   #denote the cluster with minimum mean velocity as ground scatter
	gsfg_max_vel = np.argmax(median_vels)   #denote the cluster with maxmum mean velocity as ionospheric scatter

	for i in range(n_classes):
		if (i != gsfg_min_vel and i != gsfg_max_vel):
			gsfg_undetermined = i   #denote the third cluster as indeterminate scatter

	print gsfg_min_vel
	print gsfg_max_vel


	new_gsflg = []
	tnum_scatter = 0
	for i in range(len(num_scatter)):
		new_gsflg.append(Z[tnum_scatter:(tnum_scatter+num_scatter[i])].tolist())
		tnum_scatter += num_scatter[i]

    #ipdb.set_trace()
	data_dict['gsflg'] = new_gsflg
    #print len(new_gsflg)



    #calculate GS/IS identification accuracy##############################################################################################
	num_true_trad_gs = len(np.where(((gs_flg == 1) | (gs_flg == -1)) & (emp_gsflg == 1))[0])
	num_true_trad_is = len(np.where(((gs_flg == 0)) & (emp_gsflg == 0))[0])

	num_emp = len(emp_gsflg)
	accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.
	print 'The GS/IS identification accurary of traditional method is {:3.2f}%'.format(accur_tra)


	num_true_gmm_gs1 = len(np.where((Z == gsfg_min_vel) & (emp_gsflg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
	num_true_gmm_is1 = len(np.where(((Z == gsfg_max_vel) | (Z == gsfg_undetermined)) & (emp_gsflg == 0))[0])

	num_true_gmm_gs2 = len(np.where(((Z == gsfg_min_vel) | (Z == gsfg_undetermined)) & (emp_gsflg == 1))[0])
	num_true_gmm_is2 = len(np.where((Z == gsfg_max_vel) & (emp_gsflg == 0))[0]) #Assuming the IS is the cluster with maximum median velocity

	accur_gmm1 = float(num_true_gmm_gs1+num_true_gmm_is1)/num_emp*100.
	print 'Assuming the GS is the cluster with minimum median velocity and the IS is the remaining two clusters, the GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm1)

	accur_gmm2 = float(num_true_gmm_gs2+num_true_gmm_is2)/num_emp*100.
	print 'Assuming the IS is the cluster with maximum median velocity and the GS is the remaining two clusters, the GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm2)

	accur_gmm = max(accur_gmm1,accur_gmm2)
    #calculate GS/IS identification accuracy###########################################################################################

	plot_rti(rad, stm, etm, data_dict,'velocity',gsct=True,gsfg_min_vel=gsfg_min_vel,fig_num=3,title_str='Gaussian Mixture Model Results')
    #GMM and RTI plot#################################################################################################################################    




    #scatter plot####################################################################################################################	
	cm = plt.cm.get_cmap('coolwarm')
	alpha = 1.0
	size = 1
	marker = 's'
	fig4 = plt.figure(figsize=(10,8))

	ax1 = plt.subplot(311)
	plt.scatter(emp_time[emp_gsflg == 1], emp_gate[emp_gsflg == 1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
	plt.scatter(emp_time[emp_gsflg == 0], emp_gate[emp_gsflg == 0],s=size,c='red',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red
	#plt.scatter(emp_time[emp_gsflg == -1], emp_gate[emp_gsflg == -1],s=size,c='blue',marker=marker, alpha=alpha)  #plot the undertermined scatter as blue
	ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	ax1.set_xlabel('Time UT')
	ax1.set_xlim([stm,etm])
	ax1.set_ylabel('Range gate')
	ax1.set_title('Empirical Model Results based on Burrell et al. 2015')
    
	ax2 = plt.subplot(312)
	plt.scatter(time[gs_flg == 1], gate[gs_flg == 1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
	plt.scatter(time[gs_flg == 0], gate[gs_flg == 0],s=size,c='red',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red
	#the indeterminate updated gflg (-1) was original ground scatter in traditional method when using the emp_data_dict
	plt.scatter(time[gs_flg == -1], gate[gs_flg == -1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) 
	ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #ax1.set_xlabel('Time UT')
	ax2.set_xlim([stm,etm])
	ax2.set_ylabel('Range gate')
	ax2.set_title('Traditional Model Results based on Blanchard et al. 2009 with an Accuracy of {:3.2f}%'.format(accur_tra))



	ax3 = plt.subplot(313)
	plt.scatter(time[Z == gsfg_min_vel], gate[Z == gsfg_min_vel],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm) #plot ground scatter as blue
	plt.scatter(time[Z == gsfg_max_vel], gate[Z == gsfg_max_vel],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)  #plot ionospheric scatter as red
    #plot the third scatter (E region/meteor scatter or noise, sometimes GS) as blue
	if accur_gmm1 > accur_gmm2:
		plt.scatter(time[Z == gsfg_undetermined], gate[Z == gsfg_undetermined],s=size,c='red',marker=marker,alpha = alpha, cmap=cm) 
	else:
		plt.scatter(time[Z == gsfg_undetermined], gate[Z == gsfg_undetermined],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm) 
	ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	ax3.set_xlabel('Time UT')
	ax3.set_xlim([stm,etm])
	ax3.set_ylabel('Range gate')
	ax3.set_title('Gaussian Mixture Model Results with an Accuracy of {:3.2f}%'.format(accur_gmm))

	fig4.tight_layout()
	fig4.savefig('Fig4.png')
    #scatter plot#######################################################################################################################


	#plt.show()

def read_from_updated_db(rad, stm, etm, baseLocation="../Data/"):
    
	""" reads the data from db instead of files
	"""
    #baseLocation="/Users/xueling/data/sqlite3/"
    # make a db connection
	dbName = rad + ".db"
	conn = sqlite3.connect(baseLocation + dbName, detect_types=sqlite3.PARSE_DECLTYPES)
	cur = conn.cursor()
    
	command = "SELECT * FROM {tb}\
		   WHERE time BETWEEN '{stm}' AND '{etm}' \
    		   ORDER BY time".\
		   format(tb="sd_table_"+rad, stm=stm, etm=etm)
    
	cur.execute(command)
	rws = cur.fetchall()

	if rws:
		data_dict = dict()

        #We'll use the following parameters (or features) to do the clustering or predictions        
		data_dict['datetime'] = [x[18] for x in rws]                #datetime
		data_dict['beam'] = [x[0] for x in rws]                     #beam number  (dimentionless)
		data_dict['nrang'] = [x[10] for x in rws]
		data_dict['num_scatter'] = [x[13] for x in rws]        #number of scatter return in one beam at one scan (dimentionless)
		data_dict['frequency'] = [x[5] for x in rws]                     #radar transmited frequency [MHz]


		data_dict['power'] = [json.loads(x[15]) for x in rws]    #returen signal power [dB]
		data_dict['velocity'] = [json.loads(x[19]) for x in rws]    #Doppler velocity [m/s]
		data_dict['width'] = [json.loads(x[22]) for x in rws]       #spectral width   [m/s]
		data_dict['gate'] = [json.loads(x[6]) for x in rws]    #range gate (dimentionless)
		data_dict['gsflg'] = [json.loads(x[7]) for x in rws]
		data_dict['hop'] = [json.loads(x[8]) for x in rws]
		data_dict['elevation'] = [json.loads(x[2]) for x in rws]                #elevation angle [degree]

	return data_dict

"""
Plot our algorithm results

"""
def plot_rti(rad, stm, etm, data_dict,plot_param,gsct=False,gsfg_min_vel=0,fig_num=2,title_str=' '):
    # Now let's plot all data.
    rmax = data_dict['nrang'][0] #number of range gate, usually = 75 or 110
    tcnt = len(data_dict['datetime'])

    x = date2num(data_dict['datetime'])
    y = np.linspace(0, rmax, rmax + 1)
    data = np.zeros((tcnt, rmax)) * np.nan
    tnum_scatter = 0

    for i in range(tcnt):
        for j in range(len(data_dict['gate'][i])):
            #print tnum_scatter+j
            if (not gsct and data_dict['gsflg'][i][j] == 0):
                data[i][data_dict['gate'][i][j]] = data_dict['velocity'][i][j]
            elif (not gsct and (data_dict['gsflg'][i][j] == 1 or data_dict['gsflg'][i][j] == -1)):
                data[i][data_dict['gate'][i][j]] = -100000.
            elif (gsct and data_dict['gsflg'][i][j] == gsfg_min_vel):
                data[i][data_dict['gate'][i][j]] = -100000.
            elif (gsct and data_dict['gsflg'][i][j] != gsfg_min_vel):
                data[i][data_dict['gate'][i][j]] = data_dict['velocity'][i][j]

    X, Y = np.meshgrid(x, y)
    Zm = np.ma.masked_where(np.isnan(data.T), data.T)
    #Zm = np.ma.masked_where(np.isnan(data[:tcnt][:].T), data[:tcnt][:].T)
    # Set colormap so that masked data (bad) is transparent.

    cmap, norm, bounds = utils.plotUtils.genCmap(plot_param, [-200,200],
                                                 colors = 'lasse',
                                                 lowGray = False)
    
    cmap.set_bad('w', alpha=0.0)
    
    pos=[.1, .1, .76, .72]
    rti_fig = plt.figure(fig_num,figsize=(8, 4))
    ax = rti_fig.add_axes(pos)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([stm,etm])
    ax.set_ylabel('Range gate')
    ax.set_title(title_str+' '+stm.strftime("%d %b %Y") + ' ' + rad.upper())
    pcoll = ax.pcolormesh(X, Y, Zm, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)
    
    # Draw the colorbar.
    cb = utils.drawCB(rti_fig, pcoll, cmap, norm, map_plot=0,
                      pos=[pos[0] + pos[2] + .02, pos[1], 0.02,
                           pos[3]])
                          
    l = []
    # Define the colorbar labels.
    for i in range(0, len(bounds)):
        if((i == 0 and plot_param == 'velocity') or i == len(bounds) - 1):
            l.append(' ')
            continue
        l.append(str(int(bounds[i])))
    cb.ax.set_yticklabels(l)

    # Set colorbar label.
    if(plot_param == 'velocity'):
        cb.set_label('Velocity [m/s]', size=10)
    
	plt.savefig('Fig'+str(fig_num)+'.png')
    #plt.show()



"""
Plot empirical model results

"""
def plot_rti_emp(rad, stm, etm, data_dict,plot_param,fig_num=1,title_str=' '):
    # Now let's plot all data.
    rmax = data_dict['nrang'][0] #number of range gate, usually = 75 or 110
    tcnt = len(data_dict['datetime'])

    x = date2num(data_dict['datetime'])
    y = np.linspace(0, rmax, rmax + 1)
    data = np.zeros((tcnt, rmax)) * np.nan
    tnum_scatter = 0
    gs_hops = [1.0, 2.0, 3.0]
    is_hops = [0.5, 1.5, 2.5]
    
    # Classify data as GS or IS by hop
    for i in range(tcnt):
        for j in range(len(data_dict['hop'][i])):
            #print tnum_scatter+j
            if data_dict['hop'][i][j] in is_hops:
                data[i][data_dict['gate'][i][j]] = data_dict['velocity'][i][j]
            elif data_dict['hop'][i][j] in gs_hops:
                data[i][data_dict['gate'][i][j]] = -100000.


    X, Y = np.meshgrid(x, y)
    Zm = np.ma.masked_where(np.isnan(data.T), data.T)
    #Zm = np.ma.masked_where(np.isnan(data[:tcnt][:].T), data[:tcnt][:].T)
    # Set colormap so that masked data (bad) is transparent.

    cmap, norm, bounds = utils.plotUtils.genCmap(plot_param, [-200,200],
                                                 colors = 'lasse',
                                                 lowGray = False)
    
    cmap.set_bad('w', alpha=0.0)
    
    pos=[.1, .1, .76, .72]
    rti_fig = plt.figure(fig_num,figsize=(8, 4))
    ax = rti_fig.add_axes(pos)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([stm,etm])
    ax.set_ylabel('Range gate')
    ax.set_title(title_str+' '+stm.strftime("%d %b %Y") + ' ' + rad.upper())
    pcoll = ax.pcolormesh(X, Y, Zm, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)
    
    # Draw the colorbar.
    cb = utils.drawCB(rti_fig, pcoll, cmap, norm, map_plot=0,
                      pos=[pos[0] + pos[2] + .02, pos[1], 0.02,
                           pos[3]])
                          
    l = []
    # Define the colorbar labels.
    for i in range(0, len(bounds)):
        if((i == 0 and plot_param == 'velocity') or i == len(bounds) - 1):
            l.append(' ')
            continue
        l.append(str(int(bounds[i])))
    cb.ax.set_yticklabels(l)

    # Set colorbar label.
    if(plot_param == 'velocity'):
        cb.set_label('Velocity [m/s]', size=10)
    
	plt.savefig('Fig'+str(fig_num)+'.png')
    #plt.show()



rad = "sas" 
#choose a date from 2017/08/28-2017/09/27 from database 
#(not including 09/17-09/19 three days for sas data is not suitable for emprical model prediction)
stm = dt.datetime(2017,9,21,0)
etm = dt.datetime(2017,9,22,0)

gmm_dist(rad, stm, etm)

