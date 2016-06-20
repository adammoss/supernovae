from train import *
import time
import ConfigParser
import argparse
import json
import itertools as it
import os.path

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-f','--f', type=str, help='Ini filename')
	args = parser.parse_args()

	if args.f:
		filename = args.f
	else:
		filename = 'test.ini'

	if not os.path.isfile(filename):
		raise Exception('File does not exist') 

	config = ConfigParser.ConfigParser()
	config.read(filename)

	rnn_type = config.get('Network','rnn_type')
	dropout = config.getfloat('Network','dropout')
	nb_hidden = json.loads(config.get('Network','hidden_layers'))
	bidirectional = config.getboolean('Network','bidirectional')
	consensus = config.getboolean('Network','consensus')

	optimizer = config.get('Train','optimizer')
	test_fraction = config.getfloat('Train','test_fraction')
	nb_epoch = config.getint('Train','num_epochs')
	batch_size = config.getint('Train','batch_size')

	nb_augment = config.getint('Data','num_augment')
	path = config.get('Data','data')
	classifier_type = config.get('Data','classifier')

	if classifier_type == 'SN1a':
		classifier = sn1a_classifier
	elif classifier_type == '123':
		classifier = type_classifier
	elif classifier_type.lower() == 'Sub':
		classifier = subtype_classifier
	else:
		raise Exception('Incorrect classifier') 

	root_filename = config.get('Options','filename')
	plot_loss = config.getboolean('Options','plot_loss')
	save_model = config.getboolean('Options','save_model')

	with open('save/' + root_filename + '.ini', 'w') as f:
		config.write(f)

	start_time = time.time()
	result = train(batch_size=batch_size, dropout=dropout, nb_hidden=nb_hidden, path=path, test_fraction=test_fraction, 
			classifier=classifier, nb_epoch=nb_epoch, bidirectional=bidirectional, rnn_type=rnn_type, save_model=save_model, 
			plot_loss=plot_loss, filename=root_filename, optimizer=optimizer, nb_augment=nb_augment,
			consensus=consensus)
	print("--- %s seconds ---" % (time.time() - start_time))

