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
from sklearn.mixture import BayesianGaussianMixture
#from sklearn.model_selection import StratifiedKFold

from davitpy import utils


def all_model_dist(rad, stm, etm, n_classes=5, vel_threshold = 10., BGM=False):
    
    #Read data with emprical model information and RTI plot###########################################################################################
	data_dict = read_from_updated_db(rad, stm, etm)

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
        
        #plot_rti_emp(rad, stm, etm, data_dict,'velocity',fig_num=1,title_str='Empirical Model Results')
    #Read data with emprical model information and RTI plot##########################################################################################

    #Read data from database and RTI plot###########################################################################################################
	#plot_rti(rad, stm, etm, data_dict,'velocity',gsct=False,fig_num=2,title_str='Traditional Model Results')

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

	#kmeans = KMeans(init = 'k-means++', n_clusters = n_classes, n_init=100).fit(data)
        Z_kmeans = KMeans(init = 'k-means++',n_clusters =  n_classes, n_init=50).fit_predict(data)

    # source
    # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
	cov_type = 'full' # ['spherical', 'diag', 'tied', 'full']
        if not BGM:
	        estimator = GaussianMixture(n_components=n_classes, \
                                            covariance_type=cov_type, max_iter=100, \
                                            random_state=0, n_init=50,init_params = 'kmeans')
        elif BGM:
                estimator = BayesianGaussianMixture(n_components=n_classes, \
                                                    covariance_type=cov_type, max_iter=100, \
                                                    random_state=0, n_init=50, init_params='kmeans')
                
	# initialize the GMM parameters with kmean centroid
	#estimator.means_init = kmeans.cluster_centers_ #np.random.random((n_classes, D))*2.0-1.0
    # Train the other parameters using the EM algorithm.
	estimator.fit(data)

        Z_gmm = estimator.predict(data)

	median_vels_kmeans = np.zeros(n_classes)
	median_wids_kmeans = np.zeros(n_classes)
	median_vels_gmm = np.zeros(n_classes)
	median_wids_gmm = np.zeros(n_classes)
        
	for i in range(n_classes):
		median_vels_kmeans[i] = np.median(np.abs(vel[Z_kmeans == i]))
		median_wids_kmeans[i] = np.median(wid[Z_kmeans == i])
 		median_vels_gmm[i] = np.median(np.abs(vel[Z_gmm == i]))
		median_wids_gmm[i] = np.median(wid[Z_gmm == i])
                
		print median_vels_gmm[i]
		print median_vels_kmeans[i]

        gs_class_kmeans = []
        is_class_kmeans = []
        
        gs_class_gmm = []
        is_class_gmm = []
        
	#gsfg_min_vel_kmeans = np.argmin(median_vels_kmeans)   #denote the cluster with minimum mean velocity as ground scatter
	#gsfg_max_vel_kmeans = np.argmax(median_vels_kmeans)   #denote the cluster with maxmum mean velocity as ionospheric scatter
	#gsfg_min_vel_gmm = np.argmin(median_vels_gmm)   #denote the cluster with minimum mean velocity as ground scatter
	#gsfg_max_vel_gmm = np.argmax(median_vels_gmm)   #denote the cluster with maxmum mean velocity as ionospheric scatter
        
	for i in range(n_classes):
                if median_vels_gmm[i] > vel_threshold:
                        is_class_gmm.append(i)
                else:
                        gs_class_gmm.append(i)
                        
	for i in range(n_classes):
                if median_vels_kmeans[i] > vel_threshold:
                        is_class_kmeans.append(i)
                else:
                        gs_class_kmeans.append(i)

        print is_class_gmm
        print gs_class_gmm
        print is_class_kmeans
        print gs_class_kmeans
        
        gs_flg_gmm = []
        for i in Z_gmm:
                if i in gs_class_gmm:
                        gs_flg_gmm.append(1)
                elif i in is_class_gmm:
                        gs_flg_gmm.append(0)

        gs_flg_gmm = np.array(gs_flg_gmm)
        
        gs_flg_kmeans = []
        for i in Z_kmeans:
                if i in gs_class_kmeans:
                        gs_flg_kmeans.append(1)
                elif i in is_class_kmeans:
                        gs_flg_kmeans.append(0)

        gs_flg_kmeans = np.array(gs_flg_kmeans)


    #calculate GS/IS identification accuracy##############################################################################################
	num_true_trad_gs = len(np.where(((gs_flg == 1) | (gs_flg == -1)) & (emp_gsflg == 1))[0])
	num_true_trad_is = len(np.where(((gs_flg == 0)) & (emp_gsflg == 0))[0])

	num_emp = len(emp_gsflg)
	accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.
	print 'The GS/IS identification accurary of traditional method is {:3.2f}%'.format(accur_tra)

        num_true_kmeans_gs = len(np.where((gs_flg_kmeans == 1) & (emp_gsflg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
	num_true_kmeans_is = len(np.where((gs_flg_kmeans == 0) & (emp_gsflg == 0))[0])
	accur_kmeans = float(num_true_kmeans_gs+num_true_kmeans_is)/num_emp*100.
        print 'The GS/IS identification accurary of kmeans is {:3.2f}%'.format(accur_kmeans)

	num_true_gmm_gs = len(np.where((gs_flg_gmm == 1) & (emp_gsflg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
	num_true_gmm_is = len(np.where((gs_flg_gmm == 0) & (emp_gsflg == 0))[0])
	accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
        print 'The GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm)
    #calculate GS/IS identification accuracy###########################################################################################
    
	#data_dict['gsflg'] = new_gsflg_gmm
	#plot_rti(rad, beam, stm, etm, data_dict,'velocity',gsct=True,gsfg_min_vel=gsfg_min_vel,fig_num=3,title_str='Gaussian Mixture Model Results')
    #GMM and RTI plot#################################################################################################################################    



    #scatter plot####################################################################################################################	
	cm = plt.cm.get_cmap('coolwarm')
	alpha = 0.2
	size = 1
	marker = 's'
	fig4 = plt.figure(figsize=(10,6))

	ax1 = plt.subplot(411)
 	plt.scatter(emp_time[emp_gsflg == 0], emp_gate[emp_gsflg == 0],s=size,c='red',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red       
	plt.scatter(emp_time[emp_gsflg == 1], emp_gate[emp_gsflg == 1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
	#plt.scatter(emp_time[emp_gsflg == -1], emp_gate[emp_gsflg == -1],s=size,c='blue',marker=marker, alpha=alpha)  #plot the undertermined scatter as blue
	ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	#ax1.set_xlabel('Time UT')
	ax1.set_xlim([stm,etm])
	ax1.set_ylabel('Range gate')
	ax1.set_title('Empirical Model Results based on Burrell et al. 2015')
    
	ax2 = plt.subplot(412)
 	plt.scatter(time[gs_flg == 0], gate[gs_flg == 0],s=size,c='red',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red       
	plt.scatter(time[gs_flg == 1], gate[gs_flg == 1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
	#the indeterminate updated gflg (-1) was original ground scatter in traditional method when using the emp_data_dict
	plt.scatter(time[gs_flg == -1], gate[gs_flg == -1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) 
	ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        #ax2.set_xlabel('Time UT')
	ax2.set_xlim([stm,etm])
	ax2.set_ylabel('Range gate')
	ax2.set_title('Traditional Model Results based on Blanchard et al. 2009 with an Accuracy of {:3.2f}%'.format(accur_tra))

	ax3 = plt.subplot(413)
	plt.scatter(time[gs_flg_kmeans == 0], gate[gs_flg_kmeans == 0],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)
	plt.scatter(time[gs_flg_kmeans == 1], gate[gs_flg_kmeans == 1],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm)
	ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	#ax3.set_xlabel('Time UT')
	ax3.set_xlim([stm,etm])
	ax3.set_ylabel('Range gate')
	ax3.set_title('Kmeans Results with an Accuracy of {:3.2f}%'.format(accur_kmeans))

        
	ax4 = plt.subplot(414)
        plt.scatter(time[gs_flg_gmm == 0], gate[gs_flg_gmm == 0],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)  #plot ionospheric scatter as red      
	plt.scatter(time[gs_flg_gmm == 1], gate[gs_flg_gmm == 1],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm) #plot ground scatter as blue
	ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	ax4.set_xlabel('Time UT')
	ax4.set_xlim([stm,etm])
	ax4.set_ylabel('Range gate')
	ax4.set_title('Gaussian Mixture Model Results with an Accuracy of {:3.2f}%'.format(accur_gmm))

	fig4.tight_layout()
	fig4.savefig('Fig4.png')
    #scatter plot#######################################################################################################################


	plt.show()

        

def read_from_updated_db(rad, stm, etm, baseLocation="/Users/xueling/data/sqlite3/"):
    
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
    


    
def accuracy_daily(rad, stm, etm, n_classes=5, vel_threshold = 10., BGM=False):

	data_dict = read_from_updated_db(rad, stm, etm)

	gs_hops = [1.0, 2.0, 3.0]
	is_hops = [0.5, 1.5, 2.5]

	emp_gsflg = []
	emp_num_scatter = data_dict['num_scatter']

	for i in range(len(emp_num_scatter)):
		for j in range(len(data_dict['hop'][i])):
			if data_dict['hop'][i][j] in is_hops:
				emp_gsflg.append(0)
			elif data_dict['hop'][i][j] in gs_hops:
				emp_gsflg.append(1) 
 
	emp_gsflg = np.array(emp_gsflg)

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

	#kmeans = KMeans(init = 'k-means++', n_clusters = n_classes, n_init=100).fit(data)
        Z_kmeans = KMeans(init = 'k-means++',n_clusters =  n_classes, n_init=50).fit_predict(data)

	cov_type = 'full' # ['spherical', 'diag', 'tied', 'full']
        if not BGM:
	        estimator = GaussianMixture(n_components=n_classes, \
                                            covariance_type=cov_type, max_iter=100, \
                                            random_state=0, n_init=50,init_params = 'kmeans')
        elif BGM:
                estimator = BayesianGaussianMixture(n_components=n_classes, \
                                                    covariance_type=cov_type, max_iter=100, \
                                                    random_state=0, n_init=50, init_params='kmeans')
                
	# initialize the GMM parameters with kmean centroid
	#estimator.means_init = kmeans.cluster_centers_ #np.random.random((n_classes, D))*2.0-1.0
        # Train the other parameters using the EM algorithm.
	estimator.fit(data)

        Z_gmm = estimator.predict(data)

	median_vels_kmeans = np.zeros(n_classes)
	median_wids_kmeans = np.zeros(n_classes)
	median_vels_gmm = np.zeros(n_classes)
	median_wids_gmm = np.zeros(n_classes)
        
	for i in range(n_classes):
		median_vels_kmeans[i] = np.median(np.abs(vel[Z_kmeans == i]))
		median_wids_kmeans[i] = np.median(wid[Z_kmeans == i])
 		median_vels_gmm[i] = np.median(np.abs(vel[Z_gmm == i]))
		median_wids_gmm[i] = np.median(wid[Z_gmm == i])


        gs_class_kmeans = []
        is_class_kmeans = []
        
        gs_class_gmm = []
        is_class_gmm = []

        
	for i in range(n_classes):
                if median_vels_gmm[i] > vel_threshold:
                        is_class_gmm.append(i)
                else:
                        gs_class_gmm.append(i)
                        
	for i in range(n_classes):
                if median_vels_kmeans[i] > vel_threshold:
                        is_class_kmeans.append(i)
                else:
                        gs_class_kmeans.append(i)

        
        gs_flg_gmm = []
        for i in Z_gmm:
                if i in gs_class_gmm:
                        gs_flg_gmm.append(1)
                elif i in is_class_gmm:
                        gs_flg_gmm.append(0)

        gs_flg_gmm = np.array(gs_flg_gmm)
        
        gs_flg_kmeans = []
        for i in Z_kmeans:
                if i in gs_class_kmeans:
                        gs_flg_kmeans.append(1)
                elif i in is_class_kmeans:
                        gs_flg_kmeans.append(0)

        gs_flg_kmeans = np.array(gs_flg_kmeans)


    #calculate GS/IS identification accuracy##############################################################################################
	num_true_trad_gs = len(np.where(((gs_flg == 1) | (gs_flg == -1)) & (emp_gsflg == 1))[0])
	num_true_trad_is = len(np.where(((gs_flg == 0)) & (emp_gsflg == 0))[0])

	num_emp = len(emp_gsflg)
	accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.
	print 'The GS/IS identification accurary of traditional method is {:3.2f}%'.format(accur_tra)

        num_true_kmeans_gs = len(np.where((gs_flg_kmeans == 1) & (emp_gsflg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
	num_true_kmeans_is = len(np.where((gs_flg_kmeans == 0) & (emp_gsflg == 0))[0])
	accur_kmeans = float(num_true_kmeans_gs+num_true_kmeans_is)/num_emp*100.
        print 'The GS/IS identification accurary of kmeans is {:3.2f}%'.format(accur_kmeans)

	num_true_gmm_gs = len(np.where((gs_flg_gmm == 1) & (emp_gsflg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
	num_true_gmm_is = len(np.where((gs_flg_gmm == 0) & (emp_gsflg == 0))[0])
	accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
        print 'The GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm)
    #calculate GS/IS identification accuracy###########################################################################################
        
        return accur_tra, accur_kmeans, accur_gmm
        
    
def accuracy_variation_with_time(rad, BGM=False):
        accur_tras = []
        accur_kmeanss = []
        accur_gmms = []
        times = []
        stm1 = dt.datetime(2017,8,28)

        for i in range(20):
                stm = stm1+dt.timedelta(i)
                accur_tra, accur_kmeans, accur_gmm = accuracy_daily(rad,stm,stm+dt.timedelta(1), BGM=False)
                accur_tras.append(accur_tra)
                accur_kmeanss.append(accur_kmeans)
                accur_gmms.append(accur_gmm)
                times.append(stm)
                
        stm2 = dt.datetime(2017,9,20)
        for i in range(8):
                stm = stm2+dt.timedelta(i)
                accur_tra, accur_kmeans, accur_gmm = accuracy_daily(rad,stm,stm+dt.timedelta(1), BGM=False) 
                accur_tras.append(accur_tra)
                accur_kmeanss.append(accur_kmeans)
                accur_gmms.append(accur_gmm)
                times.append(stm)       

        times = date2num(times)
        print accur_tras
        print accur_kmeanss
        print accur_gmms
        
        etm = dt.datetime(2017,9,28)
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        plt.plot(times,accur_tras, c = 'black', label = 'Traditional Model')
        plt.plot(times,accur_kmeanss, c = 'blue', label = 'Kmeans')
        plt.plot(times,accur_gmms, c = 'red', label = 'GMM')
        ax.set_ylim(0, 100)
        ax.legend(loc='best')
        ax.set_ylabel('Accuracy %')
        ax.set_xlabel('Date')
	ax.set_xlim([stm1,etm])
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        
        plt.show()
        


        
rad = "sas" 
#choose a date from 2017/08/28-2017/09/27 from database 
#(not including 09/17-09/19 three days for sas data is not suitable for emprical model prediction)
stm = dt.datetime(2017,9,20)
etm = dt.datetime(2017,9,21)

all_model_dist(rad, stm, etm, n_classes=5, vel_threshold = 10., BGM=False)
#accuracy_variation_with_time(rad, BGM=False)
#accuracy_variation_with_time(rad, BGM=True)

