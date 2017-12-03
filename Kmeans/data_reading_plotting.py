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

def k_means_data_dist(rad, beam, stm, etm, data_dict):
    
    #plot_rti(data_dict,'velocity',gsct=True)
       
    gate = np.hstack(data_dict['gate'])
    vel = np.absolute(np.hstack(data_dict['velocity'])) #data_dict['velocity']
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    
    time = []
    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        time.extend(date2num([data_dict['datetime'][i]]*num_scatter[i]))
    
    time = np.array(time)
    #ipdb.set_trace()
    #gsflg = np.hstack(data_dict['gsflg'])

#print np.sum(np.array(data_dict['num_scatter']))
    
#index_valid = np.where(power > 3)
#gate = gate[index_valid]
#vel = vel[index_valid]
#wid = wid[index_valid]
#power = power[index_valid]
#gsflg = gsflg[index_valid]
    
    #fig1 = plt.figure(1,figsize=(12,12))
	#fig1.suptitle(stm.strftime("%d %b %Y")+ ' to ' + etm.strftime("%d %b %Y"), fontsize=16)

#plt.subplot(221)
#plt.scatter(vel, gate,c=gsflg)
#plt.xlabel('Velocity [m/s]')
#plt.ylabel('Range gate')

#plt.subplot(222)
#plt.scatter(wid, gate,c=gsflg)
#plt.xlabel('Spectral width [m/s]')
#plt.ylabel('Range gate')

#plt.subplot(223)
#plt.scatter(power, gate,c=gsflg)
#plt.xlabel('Power [dB]')
#plt.ylabel('Range gate')

#plt.subplot(224)
#plt.scatter(vel, wid,c=gsflg)
#plt.xlabel('Velocity [m/s]')
#plt.ylabel('Spectral width [m/s]')
#fig1.tight_layout()
	#fig1.savefig(rad+'_beam'+str(beam)+'_'+stm.strftime("%y-%m-%d")+'_scatter_plot.png')
	#plt.show()

    #need to scale data before apply kmeans
    gate_scaled = preprocessing.scale(gate)
    vel_scaled = preprocessing.scale(vel)
    wid_scaled = preprocessing.scale(wid)
    power_scaled = preprocessing.scale(power)
    time_scaled = preprocessing.scale(time)

    #data = np.column_stack((gate,vel,wid,power))
    #data = np.column_stack((vel_scaled,wid_scaled))
    data = np.column_stack((gate_scaled,vel_scaled,wid_scaled,power_scaled,time_scaled))
    #data = np.column_stack((gate_scaled,vel_scaled,wid_scaled,power_scaled))
    Z = KMeans(init = 'k-means++',n_clusters = 2).fit_predict(data)
    
    #scatter_plot(data,Z)

    new_gsflg = []
    tnum_scatter = 0
    for i in range(len(num_scatter)):
        new_gsflg.append(Z[tnum_scatter:(tnum_scatter+num_scatter[i])].tolist())
        tnum_scatter += num_scatter[i]
            
    #ipdb.set_trace()
    data_dict['gsflg'] = new_gsflg
    #print len(new_gsflg)

    plot_rti(data_dict,'velocity',gsct=True)
    

def read_from_db(rad, beam, stm, etm, baseLocation="../Data/"):

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
        datetimes, beams, nrangs, num_scatters, vel, wid, power, gate, gsflg =  [], [], [], [], [], [], [], [], []
        
        #data_dict['datetime'] = [x[15] for x in rws]                #datetime
        #data_dict['beam'] = [x[0] for x in rws]                     #beam number  (dimentionless)
        #data_dict['nrang'] = [x[8] for x in rws]
        #data_dict['num_scatter'] = [len(json.loads(x[14])) for x in rws]        #number of scatter return in one beam at one scan (dimentionless)

        #data_dict['power'] = [json.loads(x[12]) for x in rws]    #returen signal power [dB]
        #data_dict['velocity'] = [json.loads(x[16]) for x in rws]    #Doppler velocity [m/s]
        #data_dict['width'] = [json.loads(x[17]) for x in rws]       #spectral width   [m/s]
        #data_dict['gate'] = [json.loads(x[14]) for x in rws]    #range gate (dimentionless)
        #data_dict['gsflg'] = [json.loads(x[5]) for x in rws]
            
        #We'll use the following four parameters (or features) to do the clustering or predictions
        for x in rws:
            power_filter = json.loads(x[12])
            ind = np.array([i for (i,pw) in enumerate(power_filter) if pw>6])
            
            if ind.size:
                vel.append(np.array(json.loads(x[16]))[ind].tolist())
                wid.append(np.array(json.loads(x[17]))[ind].tolist())
                power.append(np.array(json.loads(x[12]))[ind].tolist())
                gate.append(np.array(json.loads(x[14]))[ind].tolist())
                gsflg.append(np.array(json.loads(x[5]))[ind].tolist())
                datetimes.append(x[15])
                beams.append(x[0])
                nrangs.append(x[8])
                num_scatters.append(len(ind))

        data_dict['velocity'] = vel                                 #Doppler velocity [m/s]
        data_dict['width'] = wid                                    #spectral width   [m/s]
        data_dict['power'] = power	                                #returen signal power [dB]
        data_dict['gate'] = gate	                                #range gate (dimentionless)
        #ground scatter flag from traditional method, 1 indicates ground scatter, 0 indicate ionospheric scatter
        data_dict['gsflg'] = gsflg


        data_dict['datetime'] = datetimes                #datetime
        data_dict['beam'] = beams                    #beam number  (dimentionless)
        data_dict['nrang'] = nrangs
        data_dict['num_scatter'] = num_scatters        #number of scatter return in one beam at one scan (dimentionless)


            #The following three parameters might be useful in the future for comparison with model results
            #data_dict['elevation'] = [x[2] for x in rws]                #elevation angle [degree]
            #data_dict['freq'] = [x[4] for x in rws]                     #radar transmited frequency [MHz]


	    #ipdb.set_trace()
        #plot_rti(data_dict,'velocity',gsct=True)

    #print len(vel)
    #print len(data_dict['datetime'])
    #print len(data_dict['num_scatter'])
    #print len(data_dict['velocity'])

    
    return data_dict


def plot_rti(data_dict,plot_param,gsct=False):
    # Now let's plot all data.
    rmax = data_dict['nrang'][0] #number of range gate, usually = 75 or 110
    tcnt = len(data_dict['datetime'])
    #print tcnt
    #print len(data_dict['num_scatter'])
    #print data_dict['num_scatter'][0]
    x = date2num(data_dict['datetime'])
    y = np.linspace(0, rmax, rmax + 1)
    data = np.zeros((tcnt, rmax)) * np.nan
    tnum_scatter = 0
    #print np.sum(np.array(data_dict['num_scatter']))
    #print len(data_dict['velocity']) #=116256
    for i in range(tcnt):
        for j in range(len(data_dict['gate'][i])):
            #print tnum_scatter+j
            if (not gsct or data_dict['gsflg'][i][j] == 0):
                data[i][data_dict['gate'][i][j]] = data_dict['velocity'][i][j]
            elif gsct and data_dict['gsflg'][i][j] == 1:
                data[i][data_dict['gate'][i][j]] = -100000.
    #tnum_scatter += data_dict['num_scatter'][i]

    X, Y = np.meshgrid(x, y)
    Zm = np.ma.masked_where(np.isnan(data[:tcnt][:].T), data[:tcnt][:].T)
    # Set colormap so that masked data (bad) is transparent.

    cmap, norm, bounds = utils.plotUtils.genCmap(plot_param, [-200,200],
                                                 colors = 'lasse',
                                                 lowGray = False)
    
    cmap.set_bad('w', alpha=0.0)
    
    pos=[.1, .1, .76, .72]
    rti_fig = plt.figure(figsize=(11, 6))
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


    plt.show()


def scatter_plot(data,Z):
    fig1 = plt.figure(figsize=(12,12))

    plt.subplot(111)
    plt.scatter(data[:,0], data[:,1],c=Z)
    plt.xlabel('Scaled Velocity')
    plt.ylabel('Scaled Spectral width')

    plt.subplot(221)
    plt.scatter(data[:,1], data[:,0],c=Z)
    plt.xlabel('Scaled Velocity')
    plt.ylabel('Scaled Range gate')

    plt.subplot(222)
    plt.scatter(data[:,2], data[:,0],c=Z)
    plt.xlabel('Scaled Spectral width')
    plt.ylabel('Scaled Range gate')

    plt.subplot(223)
    plt.scatter(data[:,3], data[:,0],c=Z)
    plt.xlabel('Scaled Power')
    plt.ylabel('Scaled Range gate')

    plt.subplot(224)
    plt.scatter(data[:,1], data[:,2],c=Z)
    plt.xlabel('Scaled Velocity')
    plt.ylabel('Scaled Spectral width')
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(6,6))
    plt.subplot(111)
    ax2 = Axes3D(fig2, elev=48, azim=134) #, rect=[0, 0, .95, 1]
    ax2.scatter(data[:,1], data[:,0],data[:,2],c=Z)
    ax2.set_xlabel('Velocity [m/s]')
    ax2.set_ylabel('Range gate')
    ax2.set_zlabel('Spectral width [m/s]')

#fig4 = plt.figure(4,figsize=(6,6))
#plt.subplot(111)
#ax4 = Axes3D(fig4, elev=48, azim=134) #, rect=[0, 0, .95, 1]
#ax4.scatter(data[:,1], data[:,0],data[:,2],c=Z)
#ax4.set_xlabel('Scaled Velocity')
#ax4.set_ylabel('Scaled Range gate')
#ax4.set_zlabel('Scaled Spectral width')

#plt.show()




rad = 'SAS'
stm = dt.datetime(2017,9,11,0)
etm =  dt.datetime(2017,9,12,0)
beam = 7
data_dict = read_from_db(rad, beam, stm, etm)
k_means_data_dist(rad, beam, stm, etm, data_dict)

