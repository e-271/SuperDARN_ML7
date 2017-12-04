import datetime as dt
from datetime import timedelta
import logging
import os
import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import pdb
import numpy as np
import ipdb
import sqlite3
import json
import sys
from matplotlib.dates import date2num, num2date
from sklearn.cluster import KMeans
from sklearn import preprocessing
from davitpy import utils
from matplotlib.dates import DateFormatter

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold



def gmm_dist(rad, beam, stm, etm, data_dict):
    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    elev = np.hstack(data_dict['elevation'])
    gs_flg = np.hstack(data_dict['gsflg'])

           
    plot_rti(data_dict,'velocity',gsct=False,fig_num=1)
    
    date_time, time, freq = [], [], []
    
    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        #date_time.extend([data_dict['datetime'][i]]*num_scatter[i])
        time.extend(date2num([data_dict['datetime'][i]]*num_scatter[i]))
        freq.extend([data_dict['frequency'][i]]*num_scatter[i])
    
    time = np.array(time)
    freq = np.array(freq)

    alpha = 0.2
    size = 2
    marker = 's'
    fig2 = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(211)
    plt.scatter(time[gs_flg == 1], gate[gs_flg == 1],s=size,c='grey',marker=marker, alpha=alpha) #plot ground scatter as grey
    plt.scatter(time[gs_flg == 0], gate[gs_flg == 0],s=size,c='red',marker=marker, alpha=alpha)  #plot the other scatter (IS) as red
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #ax1.set_xlabel('Time UT')
    ax1.set_ylabel('Range gate')

    #need to scale data before apply kmeans
    gate_scaled = preprocessing.scale(gate)
    vel_scaled = preprocessing.scale(vel)
    wid_scaled = preprocessing.scale(wid)
    power_scaled = preprocessing.scale(power)
    time_scaled = preprocessing.scale(time)
    elev_scaled = preprocessing.scale(elev)
    freq_scaled = preprocessing.scale(freq)


    #data = np.column_stack((gate,vel,wid,power))
    #data = np.column_stack((vel_scaled,wid_scaled))
    data = np.column_stack((gate_scaled,vel_scaled,wid_scaled,\
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
    # initialize the GMM parameters in a supervised manner.
    #estimator.means_init = np.array([X_train[y_train == i].mean(axis=0))
    estimator.means_init = kmeans.cluster_centers_ #np.random.random((n_classes, D))*2.0-1.0
    # Train the other parameters using the EM algorithm.
    estimator.fit(data)
    Z = estimator.predict(data)

    mean_vels = np.zeros(n_classes)
    mean_wids = np.zeros(n_classes)
    for i in range(n_classes):
        mean_vels[i] = np.mean(np.abs(vel[Z == i]))
        mean_wids[i] = np.mean(wid[Z == i])
        print mean_vels[i]
        print mean_wids[i]

    gsfg_min_vel = np.argmin(mean_vels)   #denote the cluster with minimum mean velocity as ground scatter
    gsfg_max_vel = np.argmax(mean_vels)   #denote the cluster with maxmum mean velocity as ionospheric scatter
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

    
    ax2 = plt.subplot(212)
    plt.scatter(time, gate,s=size,c='blue',marker=marker,alpha = alpha) #plot the third scatter (E region/meteor scatter or noise?) as blue
    plt.scatter(time[Z == gsfg_min_vel], gate[Z == gsfg_min_vel],s=size,c='grey',marker=marker,alpha = alpha) #plot ground scatter as grey
    plt.scatter(time[Z == gsfg_max_vel], gate[Z == gsfg_max_vel],s=size,c='red',marker=marker,alpha = alpha)  #plot ionospheric scatter as red
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax2.set_xlabel('Time UT')
    ax2.set_ylabel('Range gate')
    fig2.tight_layout()

    plot_rti(data_dict,'velocity',gsct=True,gsfg_min_vel=gsfg_min_vel,fig_num=3)

    #scatter_plot(data,Z)
    plt.show()

def read_from_db(rad, beam, stm, etm, baseLocation="/Users/xueling/data/sqlite3/"):
    
    """ reads the data from db instead of files
        """
    
    # make a db connection
    dbName = rad + ".db"
    conn = sqlite3.connect(baseLocation + dbName, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()
    
    command = "SELECT * FROM {tb}\
    WHERE times BETWEEN '{stm}' AND '{etm}' \
    AND beam = '{beam}' \
    ORDER BY times".\
        format(tb='SD_DATA', stm=stm, etm=etm, beam=beam)
    
    cur.execute(command)
    rws = cur.fetchall()

#ipdb.set_trace()
    if rws:
        data_dict = {}
        datetimes, beams, nrangs, num_scatters, elev, freq  =  [], [], [], [], [], []
        vel, wid, power, gate, gsflg =  [], [], [], [], []
        
        #We'll use the following parameters (or features) to do the clustering or predictions
        for x in rws:
            power_filter = json.loads(x[12])
            gate_filter = json.loads(x[14])
            #ind = np.array([i for i, (pw,gt) in enumerate(zip(power_filter,gate_filter)) if (pw > 6 and gt > 13)])  #remove data with power <= 6 or gate <= 13
            ind = np.array([i for (i,pw) in enumerate(power_filter) if pw > 6])  #remove data with power <= 6

            if ind.size:
                vel.append(np.array(json.loads(x[16]))[ind].tolist())
                wid.append(np.array(json.loads(x[17]))[ind].tolist())
                power.append(np.array(json.loads(x[12]))[ind].tolist())
                gate.append(np.array(json.loads(x[14]))[ind].tolist())
                gsflg.append(np.array(json.loads(x[5]))[ind].tolist())
                elev.append(np.array(json.loads(x[2]))[ind].tolist())
                
                datetimes.append(x[15])
                beams.append(x[0])
                nrangs.append(x[8])
                
                freq.append(x[4])
                num_scatters.append(len(ind))

        data_dict['velocity'] = vel                                 #Doppler velocity [m/s]
        data_dict['width'] = wid                                    #spectral width   [m/s]
        data_dict['power'] = power                                    #returen signal power [dB]
        data_dict['gate'] = gate                                    #range gate (dimentionless)
        #ground scatter flag from traditional method, 1 indicates ground scatter, 0 indicate ionospheric scatter
        data_dict['gsflg'] = gsflg
        
        
        data_dict['datetime'] = datetimes                #datetime
        data_dict['beam'] = beams                        #beam number  (dimentionless)
        data_dict['nrang'] = nrangs
        data_dict['num_scatter'] = num_scatters          #number of scatter return in one beam at one scan (dimentionless)
        
        #The following three parameters might be useful in the future for comparison with model results
        data_dict['elevation'] = elev               #elevation angle [degree]
        data_dict['frequency'] = freq                    #radar transmited frequency [MHz]


    return data_dict


def plot_rti(data_dict,plot_param,gsct=False,gsfg_min_vel=0,fig_num=1):
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
            elif (not gsct and data_dict['gsflg'][i][j] == 1):
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
    ax.set_ylabel('Range gate')
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
    
    
    #plt.show()


def scatter_plot(data,Z):
    alpha = 0.3
    
    fig1 = plt.figure(figsize=(12,12))
    
    plt.subplot(111)
    plt.scatter(data[:,0], data[:,1],c=Z,alpha = alpha)
    plt.xlabel('Scaled Velocity')
    plt.ylabel('Scaled Spectral width')
    
    plt.subplot(221)
    plt.scatter(data[:,1], data[:,0],c=Z,alpha = alpha)
    plt.xlabel('Scaled Velocity')
    plt.ylabel('Scaled Range gate')
    
    plt.subplot(222)
    plt.scatter(data[:,2], data[:,0],c=Z,alpha = alpha)
    plt.xlabel('Scaled Spectral width')
    plt.ylabel('Scaled Range gate')
    
    plt.subplot(223)
    plt.scatter(data[:,3], data[:,0],c=Z,alpha = alpha)
    plt.xlabel('Scaled Power')
    plt.ylabel('Scaled Range gate')
    
    plt.subplot(224)
    plt.scatter(data[:,1], data[:,2],c=Z,alpha = alpha)
    plt.xlabel('Scaled Velocity')
    plt.ylabel('Scaled Spectral width')
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize=(6,6))
    plt.subplot(111)
    ax2 = Axes3D(fig2) #, rect=[0, 0, .95, 1] , elev=48, azim=134
    ax2.scatter(data[:,1], data[:,0],data[:,2],c=Z,alpha = alpha)
    ax2.set_xlabel('Velocity [m/s]')
    ax2.set_ylabel('Range gate')
    ax2.set_zlabel('Spectral width [m/s]')
    
    fig3 = plt.figure(figsize=(6,6))
    plt.subplot(111)
    ax2 = Axes3D(fig3) #, rect=[0, 0, .95, 1], elev=48, azim=134
    ax2.scatter(data[:,6], data[:,0],data[:,1],c=Z,alpha = alpha)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Range gate')
    ax2.set_zlabel('Velocity [m/s]')




rad = 'SAS'
#stm = dt.datetime(2017,9,20,0)
#etm =  dt.datetime(2017,9,21,0)
stm = dt.datetime(2017,9,11,0)
etm =  dt.datetime(2017,9,12,0)

beam = 7
data_dict = read_from_db(rad, beam, stm, etm)
gmm_dist(rad, beam, stm, etm, data_dict)

