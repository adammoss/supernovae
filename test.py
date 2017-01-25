#from dataset import load_data
#sn1a_classifier = {'1':0, '2':1, '3':1, '21':1, '22': 1, '23':1, '32': 1, '33':1}
#(X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), (length_train, length_test, sequence_len, output_dim, nb_classes) = load_data(path="data/unblind_nohostz", test_fraction=0.5, classifier=sn1a_classifier, nb_augment=1, detection="Both")

import glob
import json
import numpy as np
def analyse(filename):

	results = []
	maps = []

	#for f in glob.glob('arch/*.txt'):
	#SN1a and 123
	for f in ['arch/lstm_0.5_USUCTL.txt', 'arch/lstm_0.5_RBAFYR.txt', 'arch/lstm_0.75_IEXIOJ.txt', 'arch/lstm_0.75_YZIMTD.txt', 'arch/lstm_0.948_HTDJUB.txt', 'arch/lstm_0.948_HHGQPA.txt', 'arch/lstm_0.5_ZQVVVM.txt', 'arch/lstm_0.5_QDJMVX.txt', 'arch/lstm_0.948_WQQXYR.txt', 'arch/lstm_0.948_MOVTRG.txt']:
	#SN1a and 123 early epoch
	#for f in ['arch/info_0.5_OMEQFR.txt', 'arch/info_0.5_NVMIOZ.txt', 'arch/info_0.5_WSXTAT.txt', 'arch/info_0.948_KUDLXC.txt', 'arch/info_0.948_KDXFCS.txt', 'arch/info_0.948_PFNMKJ.txt', 'arch/info_0.5_CLKQZA.txt', 'arch/info_0.948_RSOVMJ.txt']:
		print f
		with open(f) as data_file:   
			p = json.load(data_file)
			if 'auc' in p:
				try:
					results.append([np.mean(p['f1']), p['classifier'][1], p, f])
					maps.append(p['classifier'][1])
				except:
					results.append([np.mean(p['f1']), p['mapping'][1], p, f])
					maps.append(p['mapping'][1])
	for m in list(set(maps)):

		map_results = [r for r in results if r[1] == m]
		sorted_results = sorted(map_results, key=lambda x: -x[0])

		print('classifier: ', m)

		with open("arch/output_" + m + ".tex", "w") as f:
			f.write("\\documentclass{article}\n")
			f.write("\usepackage[a3paper,margin=1in,landscape]{geometry}\n")
			f.write("\\begin{document}\n")
			f.write("\\begin{tabular}{|" + " | ".join(["c"] * 13) + "|} \hline \n")
			f.write(" & ".join(['File', 'AUC', 'Accuracy', 'F1', 'Purity', 'Completeness', 'Test fraction', 'z', 'Class', 'h', 'B', 'Dropout', 'Detection']) + " \\\\ \hline \n")
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
				prec_mean = [ "{:.3f}".format(np.mean(prec[:,i])) for i in range(0,prec.shape[1])]
				prec_std = [ "{:.3f}".format(np.std(prec[:,i])) for i in range(0,prec.shape[1])]
				rec_mean = [ "{:.3f}".format(np.mean(rec[:,i])) for i in range(0,prec.shape[1])]
				rec_std = [ "{:.3f}".format(np.std(rec[:,i])) for i in range(0,prec.shape[1])]
				prec_write = ''
				for i in xrange(len(prec_mean)):
					prec_write += "$"+prec_mean[i]+" \pm "+prec_std[i]+"$"
					prec_write += ", "
				rec_write = ''
				for i in xrange(len(rec_mean)):
					rec_write += "$"+rec_mean[i]+" \pm "+rec_std[i]+"$"
					rec_write += ", "
				#prec_write [ prec_mean[i] + " \pm "
				#prec_mean = "$ {:.3f}".format(np.mean(prec))
				#prec_std = "$ {:.3f}".format(np.std(prec))
				#rec_mean = "$ {:.3f}".format(np.mean(rec))
				#rec_std = "$ {:.3f}".format(np.std(rec))
				#prec_write = prec_mean + ' \pm ' + prec_std
				#rec_write = rec_mean + ' \pm ' + rec_std
				test_frac = "{:.3f}".format(p['test_fraction'])
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
				if 'detection' in p.keys():
					f.write(" & ".join([row[3].split('_')[2].replace('.txt', ''), auc, acc, f1, prec_write, rec_write, test_frac, host, p['rnn'], str(p['layers']), bi, str(p['dropout']), p['detection']]) + " \\\\\n")
					print(" & ".join([row[3].split('_')[2].replace('.txt', ''), auc, acc, f1, test_frac, host, p['rnn'], str(p['layers']), bi, str(p['dropout']), p['detection']] + prec_mean + rec_mean))
				else:
					f.write(" & ".join([row[3].split('_')[2].replace('.txt', ''), auc, acc, f1, prec_write, rec_write, test_frac, host, p['rnn'], str(p['layers']), bi, str(p['dropout']), 'All']) + " \\\\\n")
					print(" & ".join([row[3].split('_')[2].replace('.txt', ''), auc, acc, f1, test_frac, host, p['rnn'], str(p['layers']), bi, str(p['dropout']), 'All'] + prec_mean + rec_mean))
			f.write("\\end{tabular}\n")
			f.write("\\end{document}\n")

analyse('')
