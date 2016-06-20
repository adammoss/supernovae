from train import *
import itertools as it
from multiprocessing import Pool
import json
import numpy as np
from random import choice
from string import ascii_uppercase
import glob
import argparse

nb_rand = 5
root_filename = 'info'

def analyse(filename):

	results = []
	maps = []

	for f in glob.glob('arch/'+filename+'*.txt'):
		with open(f) as data_file:   
			p = json.load(data_file)
			if 'auc' in p:
				try:
					results.append([np.mean(p['auc']), p['classifier'][1], p, f])
					maps.append(p['classifier'][1])
				except:
					results.append([np.mean(p['auc']), p['mapping'][1], p, f])
					maps.append(p['mapping'][1])

	for m in list(set(maps)):

		map_results = [r for r in results if r[1] == m]
		sorted_results = sorted(map_results, key=lambda x: -x[0])

		print('classifier: ', m)

		with open("arch/"+filename+"_"+m+".tex", "w") as f:
			f.write("\\begin{tabular}{|" + " | ".join(["c"] * 9) + "|} \hline \n")
			f.write(" & ".join(['AUC', 'Accuracy', 'F1', 'z', 'Class', 'h', 'B']) + " \\\\ \hline \n")
			for row in sorted_results:
				p = row[2]
				acc_mean = np.mean(p['acc'])
				acc_std = np.std(p['acc'])
				auc_mean = np.mean(p['auc'])
				auc_std = np.std(p['auc'])
				f1_mean = np.mean(p['f1'])
				f1_std = np.std(p['f1'])
				prec = np.array(p['prec'])
				rec = np.array(p['rec'])
				prec_mean = [ "$ {:.3f}".format(np.mean(prec[:,i])) for i in range(0,prec.shape[1])]
				prec_std = [ "$ {:.3f}".format(np.std(prec[:,i])) for i in range(0,prec.shape[1])]
				rec_mean = [ "$ {:.3f}".format(np.mean(rec[:,i])) for i in range(0,prec.shape[1])]
				rec_std = [ "$ {:.3f}".format(np.std(rec[:,i])) for i in range(0,prec.shape[1])]
				auc = "$ {:.3f}".format(auc_mean) + ' \pm ' + "{:.3f} $".format(auc_std)
				acc = "$ {:.3f}".format(acc_mean) + ' \pm ' + "{:.3f} $".format(acc_std)
				f1 = "$ {:.2f}".format(f1_mean) + ' \pm ' + "{:.2f} $".format(f1_std)
				if p['bidirectional']:
					bi = 'T'
				else:
					bi = 'F'
				if 'No Host' in p['data']:
					host = 'F'
				else:
					host = 'T'
				f.write(" & ".join([row[3], auc, acc, f1, host, p['rnn'], str(p['layers']), bi, str(p['dropout'])]) + " \\\\\n")
				print(" & ".join([row[3], auc, acc, f1, host, p['rnn'], str(p['layers']), bi, str(p['dropout'])] + prec_mean + rec_mean))
			f.write("\\end{tabular}")

def test_arch(p):

	acc = []
	auc = []
	prec = []
	rec = []
	f1 = []
	plot_data = []

	filename = ''.join(choice(ascii_uppercase) for i in range(6))

	with open('arch/'+root_filename+'_'+str(p['test_fraction'])+'_'+filename+'.txt', 'w') as outfile:
		json.dump(p, outfile)

	for i in range(0,nb_rand):
		result = train(batch_size=p['batch_size'], dropout=p['dropout'], nb_hidden=p['layers'], path=p['data'][0], test_fraction=p['test_fraction'], 
			classifier=p['classifier'][0], nb_epoch=p['nb_epoch'], bidirectional=p['bidirectional'], rnn_type=p['rnn'], save_model=p['save_model'], 
			plot_loss=p['plot_loss'], filename=root_filename+'_'+filename, plot_data=plot_data, optimizer=p['optimizer'], nb_augment=p['nb_augment'])
		acc.append(result[0])
		auc.append(result[1])
		prec.append(result[2].tolist())
		rec.append(result[3].tolist())
		f1.append(result[4])
		plot_data = result[5]

		p['acc'] = acc
		p['auc'] = auc
		p['prec'] = prec
		p['rec'] = rec
		p['f1'] = f1

		with open('arch/'+root_filename+'_'+str(p['test_fraction'])+'_'+filename+'.txt', 'w') as outfile:
			json.dump(p, outfile)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-na','--na', type=int, help='Number of augmentations')
	parser.add_argument('-np','--np', type=int, help='Number of processors')
	parser.add_argument('-nr','--nr', type=int, help='Number of random runs')
	parser.add_argument('-ne','--ne', type=int, help='Number of epochs')
	parser.add_argument('-b','--b', type=int, help='Batch size')
	parser.add_argument('-r','--r', type=str, help='Root filename')
	parser.add_argument('-l','--l', type=bool, help='Plot loss')
	parser.add_argument('-s','--s', type=bool, help='Save model')
	parser.add_argument('-o','--o', type=str, help='Optimizer')

	args = parser.parse_args()

	if args.na:
		nb_augment = args.na
	else:
		nb_augment = 5

	if args.np:
		nb_proc = args.np
	else:
		nb_proc = 1

	if args.nr:
		nb_rand = args.nr

	if args.ne:
		nb_epoch = args.ne
	else:
		nb_epoch = 200

	if args.b:
		batch_size = args.b
	else:
		batch_size = 10

	if args.r:
		root_filename = args.r

	if args.o:
		optimizer = args.o
	else:
		optimizer = 'adam'  

	if args.l:
		plot_loss = args.l
	else:
		plot_loss = True

	if args.s:
		save_model = args.s
	else:
		save_model = False

	p = {}
	p['rnn'] = ['LSTM']
	p['layers'] = [[4], [16, 16]]
	p['data'] = [["data/unblind_nohostz", 'No Host'], ["data/unblind_hostz", 'Host']]
	p['bidirectional'] = [False, True]
	p['dropout'] = [0.5] 
	p['classifier'] = [[sn1a_classifier, 'SN1a'], [type_classifier, '123']]
	p['test_fraction'] = [0.5, 0.948]
	p['batch_size'] = [batch_size]
	p['nb_epoch'] = [nb_epoch]
	p['plot_loss'] = [plot_loss]
	p['save_model'] = [save_model]
	p['optimizer'] = [optimizer]
	p['nb_augment'] = [nb_augment]

	varNames = sorted(p)
	combinations = [dict(zip(varNames, prod)) for prod in it.product(*(p[varName] for varName in varNames))]

	if nb_proc > 1:
		pool = Pool(nb_proc)
		pool.map(test_arch, combinations)
	else:
		for c in combinations:
			test_arch(c)
