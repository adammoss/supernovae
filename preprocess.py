import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import gzip
import scipy.interpolate as si
from itertools import groupby
import random
import sys
import argparse

flux_norm = 1.0
time_norm = 1.0
position_norm = 1.0
grouping = 1

key_types = {'Ia':1, 'II': 2, 'Ibc': 3, 'IIn': 21, 'IIP': 22, 'IIL': 23, 'Ib': 32, 'Ic': 33}

def index_min(values):
	'''
	Return the index of an array.
	* values is an array (intended to be times)
	- Used in time_collector() for grouping times
	- Used in parser_spline() for placing flux errors at the correct time in the time sequence
	'''
	return min(xrange(len(values)),key=values.__getitem__)

def spline(arr,t):
	'''
	Returns the interpolated spline value of the fluxes at a given time. If the length of the
	array is too short for the spline routine it pads the values so that spline interpolation
	can be carried out.
	* arr is an array with arr = [[times],[fluxes],[flux errors]]
	* t is array of grouped times
	- Used in parser_spline() to create the spline of the fluxes
	''' 
	if (len(arr[0]) < 4):
		if (len(arr[0])==0):
			arr[0] = [t[0],int((t[-1]-t[0])/2),t[-1]]
			arr[1] = [0,0,0] 
			arr[2] = [0,0,0] 
		if (len(arr[0])==1):
			arr[0] = [t[0],arr[0][0],t[-1]]
			arr[1] = [arr[1][0],arr[1][0],arr[1][0]] 
			arr[2] = [arr[2][0],arr[2][0],arr[2][0]] 
		spline = si.InterpolatedUnivariateSpline(arr[0], arr[1],k=1)
	else:
		spline = si.InterpolatedUnivariateSpline(arr[0], arr[1])	
	return spline

def time_collector(arr,frac=grouping):
	'''
	Returns the an array of average times about clustered observation times. Default grouping is
	for times on the order of 1 day, although this is reduced if there are too many observations
	in that time. Also returns the index of the indices of the closest times in each flux band
	and the grouping fraction.
	* arr is an array containing all of the observation times
	* frac is the clustering scale where frac=1 is group times within a day
	* a is the array of grouped times
	- Used in parser_spline() for grouping flux errors to the nearest grouped time
	- Used in parser_augment() for grouping times from all observations
	'''
	bestclustering = True
	while bestclustering:
		a = []
		for key, group in groupby(arr, key=lambda n: n//(1./frac)):
			s = sorted(group)
			a.append(np.sum(s)/len(s)) 
		ind = []
		i = 0
		for key,group in groupby(arr, key=lambda n: n//(1./frac)):
			ind.append([])
			for j in group:
				ind[i].append(index_min(abs(j-np.array(arr))))
			i += 1
		if len([len(i) for i in ind if len(i)>4])!=0:
			frac += 0.1
		else:
			bestclustering = False
	return a,ind,frac

def create_colourband_array(ind,arr,err_arr,temp_arr,err_temp_arr):
	'''
	Returns arrays containing the all of the flux observations, all of the flux error observations
	and an option to check that times are grouped such that there is only one observation in a
	cluster of times.
	* ind is the list of indices containing the nearest grouped time for each observation
	* arr is array of all of the flux observations at all observation times
	* err_arr is the array of all of the flux error observations at all observation times
	* temp_arr is the array containing the fluxes at grouped times
	* temp_err_arr is the array containing the flux errors at grouped times
	* out is a boolean which is True if there is only one observation per grouped time and False
	  if there is more than one grouped time - the grouping factor is then reduced.
	- Used in parser_augment() to create the flux and flux error arrays at grouped times
	'''
	temp = [arr[ind[i]] for i in xrange(len(ind)) if arr[ind[i]]!=-999]
	err_temp = [err_arr[ind[i]] for i in xrange(len(ind)) if err_arr[ind[i]]!=-999]
	if len(temp)==0:
		temp_arr.append(-999)
		err_temp_arr.append(-999)
		out = True
	elif len(temp)>1:
		out = False
	else:
		temp_arr.append(temp[0])
		err_temp_arr.append(err_temp[0])
		out = True
	return temp_arr,err_temp_arr,out

def fill_in_points(arr,err_arr):
	'''
	Returns flux and flux error arrays where missing data is filled in with a random value between
	the previous and the next filled array elements. Missing intial or final data is filled in with
	the first or last non-missing data value respectively.
	* arr is the array of fluxes
	* err_arr is the array of flux errors
	- Used in parser_augment() to fill in missing data in flux and flux error arrays.
	'''
	ind = np.where(np.array(arr)!=-999)[0]
	length = len(arr)
	if len(ind)==0:
		arr = [0 for i in xrange(length)]
		err_arr = [0 for i in xrange(length)]
	else:
		for i in xrange(len(ind)-1):
			diff = ind[i+1]-ind[i]
			arr[ind[i]+1:ind[i+1]] = np.random.uniform(arr[ind[i]],arr[ind[i+1]],diff-1)
			err_arr[ind[i]+1:ind[i+1]] = np.random.uniform(err_arr[ind[i]],err_arr[ind[i+1]],diff-1)
		for i in xrange(len(arr[:ind[0]])):
			arr[i] = arr[ind[0]]
			err_arr[i] = err_arr[ind[0]]
		for i in xrange(len(arr[ind[-1]+1:])):
			arr[ind[-1]+1+i] = arr[ind[-1]]
			err_arr[ind[-1]+1+i] = err_arr[ind[-1]]
	return arr,err_arr

def parser_last(filename):
	'''
	Reads and returns supernovae data into format to be read by the neural network. Replaces missing observation
	data with previous non-missing observation data - steps in data are present.
	* filename is a string containing the path to the supernovae light curve data
	* survey is a string containing the survey name
	* snid is an integer containing the supernova ID
	* ra is a float containing the RA of the supernova
	* dec is a float containing the Dec of the supernova
	* mwebv is a float describing the dust extinction
	* hostid is an integer containing the host galaxy ID
	* hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
	* spec is an array of floats containing the redshift
	* sim_type is a string containing the supernova type
	* sim_z is a float containing the redshift of the supernova
	* obs is a sequence of arrays each element containing [time since first observation,fluxes in each colourband,flux errors in each colourband]
	- Used in __main__() to read in the data
	'''
	survey = snid = ra = dec = mwebv = hostid = hostz = spec = sim_type = sim_z = None
	obs = []
	g = r = i = z = 0
	g_error = r_error = i_error = z_error = 0
	with open(filename, 'rU') as f:
		first_obs = None
		for line in f:
			s = line.split(':')
			if len(s) > 0:
				if s[0] == 'SURVEY':
					survey = s[1].strip()
				elif s[0] == 'SNID':
					snid = int(s[1].strip())
				elif s[0] == 'SNTYPE':
					sn_type = int(s[1].strip())
				elif s[0] == 'RA':
					ra = float(s[1].split('deg')[0].strip())/position_norm
				elif s[0] == 'DECL':
					decl = float(s[1].split('deg')[0].strip())/position_norm
				elif s[0] == 'MWEBV':
					mwebv = float(s[1].split('MW')[0].strip())
				elif s[0] == 'HOST_GALAXY_GALID':
					hostid = int(s[1].strip())
				elif s[0] == 'HOST_GALAXY_PHOTO-Z':
					hostz = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
				elif s[0] == 'REDSHIFT_SPEC':
					spec = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
				elif s[0] == 'SIM_COMMENT':
					sim_type = s[1].split('SN Type =')[1].split(',')[0].strip()
				elif s[0] == 'SIM_REDSHIFT':
					sim_z = float(s[1])
				elif s[0] == 'OBS':
					o = s[1].split() 
					if first_obs is None:
						first_obs = float(o[0])
					if o[1] == 'g':
						g = float(o[3])/flux_norm
						g_error = float(o[4])/flux_norm
					elif o[1] == 'r':
						r = float(o[3])/flux_norm
						r_error = float(o[4])/flux_norm
					elif o[1] == 'i':
						i = float(o[3])/flux_norm
						i_error = float(o[4])/flux_norm
					elif o[1] == 'z':
						z = float(o[3])/flux_norm
						z_error = float(o[4])/flux_norm
					obs.append([(float(o[0]) - first_obs)/time_norm] + [g,r,i,z] + [g_error,r_error,i_error,z_error])
	return survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs

def parser_spline(filename):
	'''
	Reads and returns supernovae data into format to be read by the neural network. Flux observations are interpolated at grouped times
	and the errors are attributed to the grouped time closest to when they were actually measured.
	* filename is a string containing the path to the supernovae light curve data
	* survey is a string containing the survey name
	* snid is an integer containing the supernova ID
	* ra is a float containing the RA of the supernova
	* dec is a float containing the Dec of the supernova
	* mwebv is a float describing the dust extinction
	* hostid is an integer containing the host galaxy ID
	* hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
	* spec is an array of floats containing the redshift
	* sim_type is a string containing the supernova type
	* sim_z is a float containing the redshift of the supernova
	* obs is a sequence of arrays each element containing [time since first observation,fluxes in each colourband,flux errors in each colourband]
	- Used in __main__() to read in the data
	'''
	survey = snid = ra = dec = mwebv = hostid = hostz = spec = sim_type = sim_z = None
	obs = []
	t = []
        t_arr = []
	g_arr = [[],[],[]]
	r_arr = [[],[],[]]
	i_arr = [[],[],[]]
	z_arr = [[],[],[]]
	with open(filename, 'rU') as f:
		first_obs = None
		for line in f:
			s = line.split(':')
			if len(s) > 0:
				if s[0] == 'SURVEY':
					survey = s[1].strip()
				elif s[0] == 'SNID':
					snid = int(s[1].strip())
				elif s[0] == 'SNTYPE':
					sn_type = int(s[1].strip())
				elif s[0] == 'RA':
					ra = float(s[1].split('deg')[0].strip())/position_norm
				elif s[0] == 'DECL':
					decl = float(s[1].split('deg')[0].strip())/position_norm
				elif s[0] == 'MWEBV':
					mwebv = float(s[1].split('MW')[0].strip())
				elif s[0] == 'HOST_GALAXY_GALID':
					hostid = int(s[1].strip())
				elif s[0] == 'HOST_GALAXY_PHOTO-Z':
					hostz = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
				elif s[0] == 'REDSHIFT_SPEC':
					spec = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
				elif s[0] == 'SIM_COMMENT':
					sim_type = s[1].split('SN Type =')[1].split(',')[0].strip()
				elif s[0] == 'SIM_REDSHIFT':
					sim_z = float(s[1])
				elif s[0] == 'OBS':
					o = s[1].split()
					if first_obs is None:
						first_obs = float(o[0])
					t_arr.append((float(o[0])-first_obs)/time_norm)
					if o[1] == 'g':
						g_arr[0].append((float(o[0])-first_obs)/time_norm)
						g_arr[1].append(float(o[3])/flux_norm)
						g_arr[2].append(float(o[4])/flux_norm)
					elif o[1] == 'r':
						r_arr[0].append((float(o[0])-first_obs)/time_norm)
						r_arr[1].append(float(o[3])/flux_norm)
						r_arr[2].append(float(o[4])/flux_norm)
					elif o[1] == 'i':
						i_arr[0].append((float(o[0])-first_obs)/time_norm)
						i_arr[1].append(float(o[3])/flux_norm)
						i_arr[2].append(float(o[4])/flux_norm)
					elif o[1] == 'z':
						z_arr[0].append((float(o[0])-first_obs)/time_norm)
						z_arr[1].append(float(o[3])/flux_norm)
						z_arr[2].append(float(o[4])/flux_norm)
	g_spline = spline(g_arr,t_arr)
	r_spline = spline(r_arr,t_arr)
	i_spline = spline(i_arr,t_arr)
	z_spline = spline(z_arr,t_arr)
	t,ind,frac = time_collector(t_arr)	
	obs = [[t[i],g_spline(t[i]).tolist(),r_spline(t[i]).tolist(),i_spline(t[i]).tolist(),z_spline(t[i]).tolist(),g_arr[2][index_min(abs(g_arr[0]-t[i]))],r_arr[2][index_min(abs(r_arr[0]-t[i]))],i_arr[2][index_min(abs(i_arr[0]-t[i]))],z_arr[2][index_min(abs(z_arr[0]-t[i]))]] for i in xrange(len(t))]
	return survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs

def parser_augment(filename):
	'''
	Reads and returns supernovae data into format to be read by the neural network. Flux observations and errors are grouped by time
	and any missing information is filled in with random numbers between the previous and next non-missing array elements. This can
	be run many times to augment the data and create a larger train/test set. This is the preferred method of reading data.
	* filename is a string containing the path to the supernovae light curve data
	* survey is a string containing the survey name
	* snid is an integer containing the supernova ID
	* ra is a float containing the RA of the supernova
	* dec is a float containing the Dec of the supernova
	* mwebv is a float describing the dust extinction
	* hostid is an integer containing the host galaxy ID
	* hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
	* spec is an array of floats containing the redshift
	* sim_type is a string containing the supernova type
	* sim_z is a float containing the redshift of the supernova
	* obs is a sequence of arrays each element containing [time since first observation,fluxes in each colourband,flux errors in each colourband]
	- Used in __main__() to read in the data
	'''
	survey = snid = ra = dec = mwebv = hostid = hostz = spec = sim_type = sim_z = None
	obs = []
	with open(filename, 'rU') as f:
		first_obs = None
		for line in f:
			s = line.split(':')
			g = r = i = z = -999
			g_error = r_error = i_error = z_error = -999
			if len(s) > 0:
				if s[0] == 'SURVEY':
					survey = s[1].strip()
				elif s[0] == 'SNID':
					snid = int(s[1].strip())
				elif s[0] == 'SNTYPE':
					sn_type = int(s[1].strip())
				elif s[0] == 'RA':
					ra = float(s[1].split('deg')[0].strip())/position_norm
				elif s[0] == 'DECL':
					decl = float(s[1].split('deg')[0].strip())/position_norm
				elif s[0] == 'MWEBV':
					mwebv = float(s[1].split('MW')[0].strip())
				elif s[0] == 'HOST_GALAXY_GALID':
					hostid = int(s[1].strip())
				elif s[0] == 'HOST_GALAXY_PHOTO-Z':
					hostz = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
				elif s[0] == 'REDSHIFT_SPEC':
					spec = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
				elif s[0] == 'SIM_COMMENT':
					sim_type = s[1].split('SN Type =')[1].split(',')[0].strip()
				elif s[0] == 'SIM_REDSHIFT':
					sim_z = float(s[1])
				elif s[0] == 'OBS':
					o = s[1].split() 
					if first_obs is None:
						first_obs = float(o[0])
					if o[1] == 'g':
						g = float(o[3])/flux_norm
						g_error = float(o[4])/flux_norm
					elif o[1] == 'r':
						r = float(o[3])/flux_norm
						r_error = float(o[4])/flux_norm
					elif o[1] == 'i':
						i = float(o[3])/flux_norm
						i_error = float(o[4])/flux_norm
					elif o[1] == 'z':
						z = float(o[3])/flux_norm
						z_error = float(o[4])/flux_norm
					obs.append([(float(o[0]) - first_obs)/time_norm] + [g,r,i,z] + [g_error,r_error,i_error,z_error])
	t_arr = [obs[i][0] for i in xrange(len(obs))]
	g_arr = [obs[i][1] for i in xrange(len(obs))]
	g_err_arr = [obs[i][5] for i in xrange(len(obs))]
	r_arr = [obs[i][2] for i in xrange(len(obs))]
	r_err_arr = [obs[i][6] for i in xrange(len(obs))]
	i_arr = [obs[i][3] for i in xrange(len(obs))]
	i_err_arr = [obs[i][7] for i in xrange(len(obs))]
	z_arr = [obs[i][4] for i in xrange(len(obs))]
	z_err_arr = [obs[i][8] for i in xrange(len(obs))]
	correctplacement = True
	frac = grouping
	j = 0
	while correctplacement:
		t,index,frac = time_collector(t_arr,frac) 
		g_temp_arr = []
		g_err_temp_arr = []
		r_temp_arr = []
		r_err_temp_arr = []
		i_temp_arr = []
		i_err_temp_arr = []
		z_temp_arr = []
		z_err_temp_arr = []
		tot = []
		for i in xrange(len(index)):
			g_temp_arr,g_err_temp_arr,gfail = create_colourband_array(index[i],g_arr,g_err_arr,g_temp_arr,g_err_temp_arr)
			r_temp_arr,r_err_temp_arr,rfail = create_colourband_array(index[i],r_arr,r_err_arr,r_temp_arr,r_err_temp_arr)
			i_temp_arr,i_err_temp_arr,ifail = create_colourband_array(index[i],i_arr,i_err_arr,i_temp_arr,i_err_temp_arr)
			z_temp_arr,z_err_temp_arr,zfail = create_colourband_array(index[i],z_arr,z_err_arr,z_temp_arr,z_err_temp_arr)
			tot.append(gfail*rfail*ifail*zfail)
		if all(tot):
			correctplacement = False
		else:
			frac += 0.1
	
	g_temp_arr,g_err_temp_arr = fill_in_points(g_temp_arr,g_err_temp_arr)
	r_temp_arr,r_err_temp_arr = fill_in_points(r_temp_arr,r_err_temp_arr)
	i_temp_arr,i_err_temp_arr = fill_in_points(i_temp_arr,i_err_temp_arr)
	z_temp_arr,z_err_temp_arr = fill_in_points(z_temp_arr,z_err_temp_arr)
	obs = [[t[i],g_temp_arr[i],r_temp_arr[i],i_temp_arr[i],z_temp_arr[i],g_err_temp_arr[i],r_err_temp_arr[i],i_err_temp_arr[i],z_err_temp_arr[i]] for i in xrange(len(t))]
	return survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs

if __name__ == '__main__':
	'''
	Program to preprocess supernovae data. Reads in all supernova data and writes it out to one file to
	be read in by the neural network training program.
	- Reads in files from ./data/SIMGEN_PUBLIC_DES/ which contains all light curve data.
	- Creates files in ./data/
	'''

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-p','--p', type=str, help='Parser type')
	parser.add_argument('-pr','--pr', type=str, help='File prefix')
	parser.add_argument('-na','--na', type=int, help='Number of augmentations')
	args = parser.parse_args()

	if args.na:
		nb_augment = args.na
	else:
		nb_augment = 5

	if args.p:
		if args.p == 'augment':
			parser = parser_augment
		elif args.p == 'spline':
			parser = parser_spline 
			nb_augment = 1
		elif args.p == 'last':
			parser = parser_last
			nb_augment = 1
		else:
			parser = parser_augment
	else:
		parser = parser_augment

	if args.pr:
		prefix = args.pr
	else:
		prefix = ''

	for i in xrange(1,nb_augment+1):

		print 'Processing augmentation: ',i

		if prefix:
			fhost = open('data/'+prefix+'_unblind_hostz_'+str(i)+'.csv', 'w')
			fnohost = open('data/'+prefix+'_unblind_nohostz_'+str(i)+'.csv', 'w')
		else:
			fhost = open('data/unblind_hostz_'+str(i)+'.csv', 'w')
			fnohost = open('data/unblind_nohostz_'+str(i)+'.csv', 'w')
		whost = csv.writer(fhost)
		wnohost = csv.writer(fnohost)
		
		sn_types = {}
		nb_sn = 0

		for f in glob.glob('data/SIMGEN_PUBLIC_DES/DES_*.DAT'):	
		
			survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs = parser(f)
			try:
				unblind = [sim_z, key_types[sim_type]]
			except:
				print 'No information for', snid
			for o in obs:
				whost.writerow([snid,o[0],ra,decl,mwebv,hostz[0]] + o[1:9] + unblind)
				wnohost.writerow([snid,o[0],ra,decl,mwebv] + o[1:9] + unblind)
			try:
				sn_types[unblind[1]] += 1
			except:
				sn_types[unblind[1]] = 0
			nb_sn += 1

		fhost.close()
		fnohost.close()
		
	print 'Num train: ', nb_sn
	print 'SN types: ', sn_types

