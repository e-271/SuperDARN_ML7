import datetime as dt
from datetime import timedelta
#from davitpy.pydarn.sdio.fetchUtils import fetch_local_files
#from davitpy.pydarn.sdio import radDataOpen, radDataReadRec
#from davitpy.pydarn.sdio import radDataPtr
#import davitpy
import logging
import os
import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import pdb
import numpy as np
import ipdb

from sklearn.cluster import KMeans
from sklearn import preprocessing


def k_means_data_dist(rad, beam, stm, etm, data_dict):


	gate = data_dict['gate']
	vel = map(abs, data_dict['velocity']) #data_dict['velocity']
	wid = data_dict['width']
	power = data_dict['power']
	gsflg = data_dict['gsflg'] 	

	fig1 = plt.figure(1,figsize=(12,12))
	#fig1.suptitle(stm.strftime("%d %b %Y")+ ' to ' + etm.strftime("%d %b %Y"), fontsize=16)

	plt.subplot(221)
	plt.scatter(vel, gate,c=gsflg)
	plt.xlabel('Velocity [m/s]')
	plt.ylabel('Range gate')

	plt.subplot(222)
	plt.scatter(wid, gate,c=gsflg)
	plt.xlabel('Spectral width [m/s]')
	plt.ylabel('Range gate')

	plt.subplot(223)
	plt.scatter(power, gate,c=gsflg)
	plt.xlabel('Power [dB]')
	plt.ylabel('Range gate')

	plt.subplot(224)
	plt.scatter(vel, wid,c=gsflg)
	plt.xlabel('Velocity [m/s]')
	plt.ylabel('Spectral width [m/s]')

	fig1.tight_layout()
	#fig1.savefig(rad+'_beam'+str(beam)+'_'+stm.strftime("%y-%m-%d")+'_scatter_plot.png')
	#plt.show()


	#need to scale data before apply kmeans
    	gate_scaled = preprocessing.scale(gate)
    	vel_scaled = preprocessing.scale(vel)
    	wid_scaled = preprocessing.scale(wid)
    	power_scaled = preprocessing.scale(power)

    	#data = np.column_stack((gate,vel,wid,power))
    	#data = np.column_stack((vel_scaled,wid_scaled))
    	data = np.column_stack((gate_scaled,vel_scaled,wid_scaled,power_scaled))


    	Z = KMeans(init = 'k-means++',n_clusters = 2).fit_predict(data)

	fig2 = plt.figure(2,figsize=(12,12))

	#plt.subplot(111)
	#plt.scatter(data[:,0], data[:,1],c=Z)
	#plt.xlabel('Scaled Velocity')
	#plt.ylabel('Scaled Spectral width')

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

	fig2.tight_layout()


	fig3 = plt.figure(3,figsize=(6,6))
	plt.subplot(111)
	ax3 = Axes3D(fig3, elev=48, azim=134) #, rect=[0, 0, .95, 1]
	ax3.scatter(vel, gate,wid,c=gsflg)
	ax3.set_xlabel('Velocity [m/s]')
    	ax3.set_ylabel('Range gate')
    	ax3.set_zlabel('Spectral width [m/s]')

	fig4 = plt.figure(4,figsize=(6,6))
	plt.subplot(111)
	ax4 = Axes3D(fig4, elev=48, azim=134) #, rect=[0, 0, .95, 1]
	ax4.scatter(data[:,1], data[:,0],data[:,2],c=Z)
	ax4.set_xlabel('Scaled Velocity')
    	ax4.set_ylabel('Scaled Range gate')
    	ax4.set_zlabel('Scaled Spectral width')


    	plt.show()



def read_from_db(rad, beam, stm, etm, baseLocation="../Data/"):

        """ reads the data from db instead of files
        """
        
        import sqlite3
        import json
        import sys 

        
        # make a db connection
        dbName = rad + ".db"
        conn = sqlite3.connect(baseLocation + dbName, detect_types=sqlite3.PARSE_DECLTYPES) 
        cur = conn.cursor()
	#choose one beam data firstly
	#beam = 10
        command = "SELECT * FROM {tb}\
                   WHERE times BETWEEN '{stm}' AND '{etm}' \
                   AND beam = '{beam}' \
                   ORDER BY times".\
                   format(tb='SD_DATA', stm=stm, etm=etm, beam=beam)


        cur.execute(command)

        rws = cur.fetchall()

        if rws:
            data_dict = {}
            vel, wid, power, gate, gsflg =  [], [], [], [], []
            data_dict['datetime'] = [x[15] for x in rws]                #datetime
            #data_dict['beam'] = [x[0] for x in rws]                     #beam number  (dimentionless)
            data_dict['num_scatter'] = [len(x[14]) for x in rws]        #number of scatter return in one beam at one scan (dimentionless)
	
            #We'll use the following four parameters (or features) to do the clustering or predictions	
            for x in rws:
		vel.extend(json.loads(x[16])) 
		wid.extend(json.loads(x[17]))
		power.extend(json.loads(x[12]))
		gate.extend(json.loads(x[14]))
		gsflg.extend(json.loads(x[5]))

            data_dict['velocity'] = vel                                 #Doppler velocity [m/s]
            data_dict['width'] = wid                                    #spectral width   [m/s]
            data_dict['power'] = power	                                #returen signal power [dB]
            data_dict['gate'] = gate	                                #range gate (dimentionless)
            #ground scatter flag from traditional method, 1 indicates ground scatter, 0 indicate ionospheric scatter 
            data_dict['gsflg'] = gsflg	

            #data_dict['velocity'] = [json.loads(x[16]) for x in rws]    #Doppler velocity [m/s]
            #data_dict['width'] = [json.loads(x[17]) for x in rws]       #spectral width   [m/s]
            #data_dict['power'] = [json.loads(x[12]) for x in rws]	#returen signal power [dB]
            #data_dict['gate'] = [json.loads(x[14]) for x in rws]	#range gate (dimentionless)

            #The following three parameters might be useful in the future for comparison with model results
            #data_dict['elevation'] = [x[2] for x in rws]                #elevation angle [degree]
            #data_dict['freq'] = [x[4] for x in rws]                     #radar transmited frequency [MHz]
            #data_dict['gsflg'] = [json.loads(x[5]) for x in rws]        

	#ipdb.set_trace()



        return data_dict


rad = 'SAS'
stm = dt.datetime(2017,9,11,0)
etm =  dt.datetime(2017,9,12,0)
beam = 7
data_dict = read_from_db(rad, beam, stm, etm)
k_means_data_dist(rad, beam, stm, etm, data_dict)

