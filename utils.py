'''
Plotting routines
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def loss_plotter(history, filename, plot_legend=True):
	'''
	Plots the test and training loss history and accuracy history after every
	epoch of training. Plots are saved in ./arch/
	* history is a list which contains the training loss and test loss and the training
	  accuracy and test accuracy which are each contained in the array. Each random
	  augmentation of train() is another dimension in the array.
	* filename is the string containing the filename to name the plot
	* plot_legend is a boolean which when True places a legend on each axis
	- Used in train(), particularly it is in on_epoch_end() from LossHistory() which is a callback
	  called at the end of each epoch of training
	'''
	plt.rc('legend',**{'fontsize':16})
	fig = plt.figure()
	gs = gridspec.GridSpec(1,2)
	gs.update(left=0.1,right=0.97,wspace=0.3)
	ax1 = plt.subplot(gs[0])
	ax1.set_xlabel('Epoch',fontsize=16)
	ax1.set_ylabel('Loss',fontsize=16,labelpad=0)
	ax1.xaxis.grid(True,'minor')
	ax1.yaxis.grid(True,'minor')
	ax1.xaxis.grid(True,'major')
	ax1.yaxis.grid(True,'major')
	plt.tick_params(axis='both',which='major',labelsize=16)
	ax2 = plt.subplot(gs[1])
	ax2.set_xlabel('Epoch',fontsize=16)
	ax2.set_ylabel('1 - Accuracy',fontsize=16,labelpad=0)
	ax2.xaxis.grid(True,'minor')
	ax2.yaxis.grid(True,'minor')
	ax2.xaxis.grid(True,'major')
	ax2.yaxis.grid(True,'major')
	plt.tick_params(axis='both',which='major',labelsize=16)
	for i in xrange(len(history)):	
		if i < len(history)-1:
			ax1.semilogy(history[i][0],color='green')
			ax1.semilogy(history[i][1],color='blue')
			ax2.semilogy(1 - np.array(history[i][2]),color='green')
			ax2.semilogy(1 - np.array(history[i][3]),color='blue')
		else:
			a, = ax1.semilogy(history[i][0],color='green')
			b, = ax1.semilogy(history[i][1],color='blue')
			c, = ax2.semilogy(1 - np.array(history[i][2]),color='green')
			d, = ax2.semilogy(1 - np.array(history[i][3]),color='blue')

	if plot_legend:
		ax1.legend([a,b],['Train','Test'],loc='best')
		ax2.legend([c,d],['Train','Test'],loc='best')

	plt.savefig('arch/'+filename+'_loss.pdf', bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)

def lightcurve_plotter(filename, parser, save=False):
	'''
	Reads in data from a light curve file and plots the light curve.
	* filename is a string containing the path to the light curve data which should be in ./data/SIMGEN_PUBLIC_DES/
	* parser is a function passed from preprocessing.py containing parser_last, parser_spline or parser_augment which
          are used to read the light curve data.
	* save is a boolean which, when True, saves the light curve plot in ./save/ . This should be False when
          lightcurve_plotter is being called from predict_plotter()
	- Used in predict_plotter() to plot the lightcurve in the first axis
	'''
	survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs = parser(filename)
	obs = np.array(obs)
	plt.rc('legend',**{'fontsize':16})
	plt.errorbar(obs[:,0], obs[:,1], yerr=obs[:,5], color='green', label='g')
	plt.errorbar(obs[:,0], obs[:,2], yerr=obs[:,6], color='red', label='r')
	plt.errorbar(obs[:,0], obs[:,3], yerr=obs[:,7], color='orange', label='i')
	plt.errorbar(obs[:,0], obs[:,4], yerr=obs[:,8], color='black', label='z')
	plt.legend()
	plt.ylabel('Flux', fontsize=16)
	plt.xlabel('Days', fontsize=16)
	plt.tick_params(axis='both', which='major', labelsize=16)
	if save:
		plt.savefig('plots/'+filename.replace('data/SIMGEN_PUBLIC_DES/','').replace('.DAT','')+'.pdf', bbox_inches='tight', pad_inches=0.1)
