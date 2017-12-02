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
# import ipdb

# from sklearn.cluster import KMeans
from sklearn import preprocessing

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from kmeans_SAS import read_from_db


def gmm_dist(rad, beam, stm, etm, data_dict):
    gate = data_dict['gate']
    vel = map(abs, data_dict['velocity']) #data_dict['velocity']
    wid = data_dict['width']
    power = data_dict['power']
    gsflg = data_dict['gsflg']

    # fig1 = plt.figure(1,figsize=(12,12))
    #fig1.suptitle(stm.strftime("%d %b %Y")+ ' to ' + etm.strftime("%d %b %Y"), fontsize=16)

    # plt.subplot(221)
    # plt.scatter(vel, gate,c=gsflg)
    # plt.xlabel('Velocity [m/s]')
    # plt.ylabel('Range gate')

    # plt.subplot(222)
    # plt.scatter(wid, gate,c=gsflg)
    # plt.xlabel('Spectral width [m/s]')
    # plt.ylabel('Range gate')

    # plt.subplot(223)
    # plt.scatter(power, gate,c=gsflg)
    # plt.xlabel('Power [dB]')
    # plt.ylabel('Range gate')

    # plt.subplot(224)
    # plt.scatter(vel, wid,c=gsflg)
    # plt.xlabel('Velocity [m/s]')
    # plt.ylabel('Spectral width [m/s]')

    # fig1.tight_layout()
    #fig1.savefig(rad+'_beam'+str(beam)+'_'+stm.strftime("%y-%m-%d")+'_scatter_plot.png')
    #plt.show()


    #need to scale data before apply kmeans
    gate_scaled = preprocessing.scale(gate)
    vel_scaled = preprocessing.scale(vel)
    wid_scaled = preprocessing.scale(wid)
    power_scaled = preprocessing.scale(power)
    time_scaled = preprocessing.scale(list(range(len(power))))

    #data = np.column_stack((gate,vel,wid,power))
    #data = np.column_stack((vel_scaled,wid_scaled))
    full_data = np.column_stack((gate_scaled, vel_scaled, wid_scaled, \
            power_scaled, time_scaled))

    # Break up the dataset into non-overlapping training (95%) and testing
    # (5%) sets.
    skf = StratifiedKFold(n_splits=20, shuffle=True)
    # Only take the first fold.
    N,D = full_data.shape # TODO UGLY, FIX!
    train_index, test_index = next(iter(skf.split(full_data, np.ones(N))))

    data = full_data[train_index, :]
    validation_data = full_data[test_index, :]

    N,D = data.shape
    # Z = KMeans(init = 'k-means++',n_clusters = 2).fit_predict(data)
    # source
    # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
    n_classes = 4
    cov_type = 'full' # ['spherical', 'diag', 'tied', 'full']
    estimator = GaussianMixture(n_components=n_classes, \
                           covariance_type=cov_type, max_iter=20, \
                           random_state=0)
    # initialize the GMM parameters in a supervised manner.
    # estimator.means_init = np.array([X_train[y_train ==i].mean(axis=0))
    estimator.means_init = np.random.random((n_classes, D))*2.0-1.0
    # Train the other parameters using the EM algorithm.
    estimator.fit(data)

    fig2 = plt.figure(2,figsize=(12,12))

    for plot_data, marker, alpha in zip([data, validation_data], ['.','x'], \
            [0.1, 0.7]):
        Z = estimator.predict(plot_data)

        #plt.subplot(111)
        #plt.scatter(plot_data[:,0], plot_data[:,1],c=Z)
        #plt.xlabel('Scaled Velocity')
        #plt.ylabel('Scaled Spectral width')

        plt.subplot(221)
        plt.scatter(plot_data[:,4], plot_data[:,0],c=Z, marker=marker,
                alpha=alpha)
        plt.xlabel('Scaled Time')
        plt.ylabel('Scaled Range gate')

        plt.subplot(222)
        plt.scatter(plot_data[:,2], plot_data[:,0],c=Z, marker=marker,
                alpha=alpha)
        plt.xlabel('Scaled Spectral width')
        plt.ylabel('Scaled Range gate')

        plt.subplot(223)
        plt.scatter(plot_data[:,3], plot_data[:,0],c=Z, marker=marker,
                alpha=alpha)
        plt.xlabel('Scaled Power')
        plt.ylabel('Scaled Range gate')

        plt.subplot(224)
        plt.scatter(plot_data[:,1], plot_data[:,2],c=Z, marker=marker,
                alpha=alpha)
        plt.xlabel('Scaled Velocity')
        plt.ylabel('Scaled Spectral width')

    fig2.tight_layout()

    plot_data = full_data
    Z = estimator.predict(plot_data)

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
    ax4.scatter(plot_data[:,4], plot_data[:,0],plot_data[:,2],c=Z)
    ax4.set_xlabel('Scaled Time')
    ax4.set_ylabel('Scaled Range gate')
    ax4.set_zlabel('Scaled Spectral width')


    plt.show()






rad = 'SAS'
stm = dt.datetime(2017,9,1,0)
etm =  dt.datetime(2017,9,2,0)
beam = 7
data_dict = read_from_db(rad, beam, stm, etm, "./")
gmm_dist(rad, beam, stm, etm, data_dict)

