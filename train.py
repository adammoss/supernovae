from __future__ import print_function
from dataset import *
from preprocess import parser_augment, parser_last, parser_spline, key_types
from utils import loss_plotter, lightcurve_plotter
import numpy as np
import pickle

def build_model(nb_hidden, nb_classes, dropout, sequence_len, output_dim, bidirectional=False, rnn_type='LSTM', activation='tanh', consensus=True): 

	'''
	Builds model of the neural network. Bidirectional layers are implemented as 2 forward 
        layers with data reversed in 1 layer to align the masks
	* nb_hidden is list of integers with the number of hidden units, e.g. [16, 16]
	* nb_classes is an integer with the number of classification classes
	* dropout is a float between 0 and 1 describing dropout probability after each hidden 
          layer (non-recurrent connections only)
	* sequence_len is an integer containing the maximum length of observations
	* output_dim is an integer containing length of vector for each step
	* bidirectional is a boolean which when True uses a bidirectional network
	* rnn_type is a string giving RNN type (either "RNN", "GRU", "LSTM")
	* activation is a string describing the activation function for the hidden layer
	* consensus is a boolean which when true mean pools the predictions from each step
	- Used in train() to build the neural network for training
	- Used in predict_probability() to build the neural network for analysis using the
	  weights saved from the training
	'''

	import keras.backend as K
	from keras.preprocessing import sequence
	from keras.models import Sequential, Model
	from keras.layers.core import Lambda, Masking
	from keras.layers import Dense, Dropout, Activation, Embedding, Merge, Input
	from keras.layers import LSTM, SimpleRNN, GRU
	from keras.layers.wrappers import TimeDistributed
	from custom_layers import MaskLambda, lambda_mask_average, MaskMerge, mask_merge

	if rnn_type.lower() == 'lstm':
		RNNLayer = LSTM
	elif rnn_type.lower() == 'gru':
		RNNLayer = GRU
	else:
		RNNLayer = SimpleRNN

	if bidirectional:

		input_a = Input(shape=(sequence_len, output_dim))
		input_b = Input(shape=(sequence_len, output_dim))
		
		x = Masking(mask_value=-1.)(input_a)
		x = RNNLayer(nb_hidden[0], return_sequences=consensus or (len(nb_hidden) > 1), activation=activation)(x)
	
		y = Masking(mask_value=-1.)(input_b)
		y = RNNLayer(nb_hidden[0], return_sequences=consensus or (len(nb_hidden) > 1), activation=activation)(y)
		
		merged = mask_merge([x, y], mode='sum')
		merged = Dropout(dropout)(merged)

		for i, h in enumerate(nb_hidden[1:]):

			x = RNNLayer(nb_hidden[0], return_sequences=consensus or (len(nb_hidden) < i+2), activation=activation)(merged)

			y = RNNLayer(nb_hidden[0], return_sequences=consensus or (len(nb_hidden) < i+2), activation=activation)(merged)
	
			merged = mask_merge([x, y], mode='sum')
			merged = Dropout(dropout)(merged)

		if consensus:
			z = TimeDistributed(Dense(nb_classes))(merged)
			z = MaskLambda(lambda_mask_average,output_shape=lambda shape: (shape[0],) + shape[2:])(z)
		else:
			z = Dense(nb_classes)(merged)

		z = Activation('softmax')(z)

		model = Model([input_a, input_b], [z])

	else:

		input_a = Input(shape=(sequence_len, output_dim))
		x = Masking(mask_value=-1.)(input_a)
		x = RNNLayer(nb_hidden[0], return_sequences=consensus or (len(nb_hidden) > 1), activation=activation)(x)
		x = Dropout(dropout)(x)

		for i, h in enumerate(nb_hidden[1:]):
			x = RNNLayer(h, return_sequences=consensus or (len(nb_hidden) < i+2), activation=activation)(x)
			x = Dropout(dropout)(x)

		if consensus:
			z = TimeDistributed(Dense(nb_classes))(x)
			z = MaskLambda(lambda_mask_average,output_shape=lambda shape: (shape[0],) + shape[2:])(z)
		else:
			z = Dense(nb_classes)(x)

		z = Activation('softmax')(z)

		model = Model([input_a], [z])

	return model

def train(batch_size=10, dropout=0.5, nb_hidden=[16, 16], path="data/unblind_nohostz", test_fraction=0.5, classifier=sn1a_classifier, nb_epoch=200, nb_augment=5, 
	bidirectional=False, rnn_type='LSTM',feedback=1, optimizer='adam', activation='tanh', save_model=False, plot_loss=False, plot_data=[], 
        filename='current', consensus=True):
	'''
	Main training module. The data is read and the neural network built and the data fed into the neural network.
	The analysis metrics are also calculated here.
	* batch_size is an integer describing the number of samples before each gradient update
	* dropout is a float between 0 and 1 describing dropout probability after each hidden 
	* nb_hidden is list of integers with the number of hidden units, e.g. [16, 16]
	* path is a string containg the path to the data created by preprocess.py
	* test_fraction is a float between 0 and 1 describing the fraction of the data to be used to test the network,
	  1-test_fraction is the fraction of the data used to train the network
	* classifier is a dictionary containing which classification is wanted (sn1a_classifier for SN1a vs. non-SN1a, 
          type_classifier for SN1 vs. SN2 vs. SN3 or subtype_classifier for SN1a vs SN1b vs ...)
	* nb_epoch is an integer containing the number of epochs to train the model
	* nb_augment is the number of random augmentations of data to use
	* bidirectional is a boolean which when True uses a bidirectional network
	* rnn_type is a string giving RNN type (either "RNN", "GRU", "LSTM")
	* feedback in an integer to control the amount of feedback writtend to stdout. 0 is none, 1 is minimal, 2 is full
	* optimizer is a string containing the optimiser type used in the model. 
          See http://keras.io/optimizers/ for more information
	* activation is a string describing the activation function for the hidden layer
	* save_model is a boolean which when True saves the weights and model details into ./save/ so that the trained 
          network can be used again
	* plot_loss is a boolean which when True saves a plot of the loss and accuracy history after each epoch
	* plot_data is a list which is updated with the loss and accuracy history at the end of training the model and
	  is appended if more than one random augmentation is used
	* filename is a string containing the save name for plots and model properties 
	* consensus is a boolean which when true mean pools the predictions from each step
	* acc is a float containing the final accuracy
	* auc is a float containing the final area under the Receiver Operating Characteristic curve
	* prec is the final precision
	* rec is the final recall
	* f1 is the SPCC figure of merit
	- Used in arch.py
	- Used in run.py
	! load_data() is in dataset.py and is used to load the data
	'''
	import keras
	from keras.callbacks import Callback

	class LossHistory(keras.callbacks.Callback):
		'''
		Function to be used during the training procedure.
		* keras.callbacks.Callback is a class in Keras see http://keras.io/callbacks/ for more information
		- Used in train()
		'''
		def __init__(self, plot_data, filename=None, plot_loss=False):
			'''
			Initialise the LossHistory class parameters.
			* plot_data is a list which is updated with the loss and accuracy history at the end of 
                          training the model and is appended if more than one random augmentation is used
			* filename is a string containing the save name for the plot 
			* plot_loss is a boolean which when True allows the loss and accuracy history to be
			  collected and plotted
			'''
			self.plot_data = plot_data
			self.plot_loss = plot_loss
			self.filename = filename

		def on_train_begin(self, logs={}):
			'''
			Initialises the test and train loss (self.loss and self.val_loss) and the test and train
                        accuracy (self.acc and self.val_acc). This is only done when the history is going to be
 			plotted.
			'''
			if self.plot_loss:
				self.loss = []
				self.val_loss = []
				self.acc = []
				self.val_acc = []

		def on_epoch_end(self, batch, logs={}):
			'''
			If history plot is going to be made then the history data is collected and passed to the
			plotting routine.
			! loss_plotter() is the plotting routine which can be found in utils.py
			'''
			if self.plot_loss:
				self.loss.append(logs.get('loss'))
				self.val_loss.append(logs.get('val_loss'))
				self.acc.append(logs.get('acc'))
				self.val_acc.append(logs.get('val_acc'))
				self.plot_data[len(self.plot_data)-1] = [self.loss, self.val_loss, self.acc, self.val_acc]
				loss_plotter(self.plot_data, self.filename)

	(X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), (length_train, length_test, sequence_len, output_dim, nb_classes) = load_data(path=path, test_fraction=test_fraction, classifier=classifier, nb_augment=nb_augment)

	model_root = 'save/' + filename
	
	if save_model:
		np.savetxt(model_root + '_training.txt', ids_train, fmt='%i')
		np.savetxt(model_root + '_test.txt', ids_test, fmt='%i')

	if feedback > 0:
		print('Build model...')

	model = build_model(nb_hidden, nb_classes, dropout, sequence_len, output_dim, bidirectional=bidirectional, rnn_type=rnn_type, activation=activation, consensus=consensus)

	model.summary()
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	if plot_loss:
		plot_data.append([])

	hist = LossHistory(plot_data, filename=filename, plot_loss=plot_loss)

	if bidirectional:
		
		model.fit([X_train, X_train_reverse], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([X_test, X_test_reverse], Y_test),callbacks=[hist])
		score, acc = model.evaluate([X_test, X_test_reverse], Y_test, batch_size=batch_size)
		Y_score = model.predict([X_test, X_test_reverse])

	else:
		model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),callbacks=[hist])
		score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
		Y_score = model.predict(X_test)
	
	Y_predict = np.argmax(Y_score,axis=1)
	Y_true = np.argmax(Y_test,axis=1)

	try:
		from sklearn.metrics import roc_auc_score, precision_score, recall_score
		auc = roc_auc_score(Y_test, Y_score, average='macro')
		prec = precision_score(Y_true, Y_predict, average=None)
		rec = recall_score(Y_true, Y_predict, average=None) 
	except:
		print('Cannot import sklearn')
		auc = 0
		prec = 0
		rec = 0

	TP = FP = TN = FN = 0

	for i in range(0, len(Y_score)):
		if Y_predict[i] == 0 and Y_true[i] == 0: TP += 1
		if Y_predict[i] == 0 and Y_true[i] != 0: FP += 1
		if Y_predict[i] != 0 and Y_true[i] != 0: TN += 1
		if Y_predict[i] != 0 and Y_true[i] == 0: FN += 1
		if feedback > 1:
			print(ids_test[i], Y_score[i], Y_predict[i], Y_test[i], Y_true[i])

	try:
		f1 = 1.0/(TP+FN)*TP**2.0/(TP+3.0*FP)
	except:
		f1 = 0

	if feedback > 0:
		print('Test score:', score)
		print('Test accuracy:', acc)
		print('AUC:', auc)
		print('Precision:', prec)
		print('Recall:', rec)
		print('SN1a confusion: ',TP, FP, TN, FN)
		print('F1: ',f1)

	if save_model:

		with open(model_root + '_results.txt', 'w') as f:
			f.write('Test score: {0}\n'.format(score))
			f.write('Test accuracy {0}\n'.format(acc))
			f.write('AUC: {0}\n'.format(auc))
			f.write('Precision {0}\n'.format(prec))
			f.write('Recall: {0}\n'.format(rec))
			f.write('SN1a confusion {0} {1} {2} {3}\n'.format(TP, FP, TN, FN))
			f.write('F1: {0}\n'.format(f1))

		with open(model_root + '_predictions.txt', 'w') as f:
			for i in range(0, len(Y_score)):
				f.write('{0} {1} {2} {3}\n'.format(ids_test[i], Y_score[i][0], Y_predict[i], Y_true[i]))

		model.save_weights(model_root + '.h5', overwrite=True)
		model_building_blocks = [nb_hidden, nb_classes, dropout, sequence_len, output_dim, bidirectional, rnn_type, activation, consensus]
		with open(model_root + '.model', 'w') as f:
			pickle.dump(model_building_blocks,f)

	return acc, auc, prec, rec, f1, plot_data

def predict_probability(data, filename, optimizer='adam'):
	'''
	Predicts probability from lightcurve.
	* data is a list of form [[[t1, ...],[t2, ...]]] such as the returned value X_train or X_test is load_data() in dataset.py
	* filename is a string containing the name of the saved model files created in train()
	* optimizer is a string containing the optimiser type used in the model. 
          See http://keras.io/optimizers/ for more information
	* probability is a list containing floats between 0 and 1 which are the probability of different classes being True, i.e.
	  [0.9,0.1] would be 90% chance of class 1 being True.
	- Used in predict_plotter() to get probability for plotting
	! pad_sequences() is in dataset.py and is used to make sure light curve data is the correct length
	'''

	model_root = 'save/' + filename
	with open(model_root + '.model', 'r') as f:
		opt = pickle.load(f)

	model = build_model(opt[0], opt[1], opt[2], opt[3], opt[4], bidirectional=opt[5], rnn_type=opt[6], activation=opt[7], consensus=opt[8])
	model.load_weights(model_root + '.h5')
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	data_copy = list(data)
	data = pad_sequences(data, dtype='float', maxlen=opt[3])
	data_post = pad_sequences(data_copy, dtype='float', padding='post', maxlen=opt[3])
	data_reverse = data_post[:,::-1,:]

	if opt[5]:
		probability = model.predict([data, data_reverse])
	else:	
		probability = model.predict(data)

	return probability

def predict_plotter(filename, model_str, parser=parser_augment):
	'''
	Plotting routine to plot the prediction probability of data being of class 1 as a function of the number of days of data observation.
	The file is saved in ./plots/
        * filename is a string containing the path to the light curve data which should be in ./data/SIMGEN_PUBLIC_DES/
	* model_str is a string containing the name of the model to load to calculated the probability.
	* parser is a function from preprocessing.py and can be either parser_last, parser_spline or parser_augment (preferred) and is used
          to read in the light curve data.
	! lightcurve_plotter() is the lightcurve plotting routine which can be found in utils.py
	'''
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec

	plt.rc('legend',**{'fontsize':16})
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2,1)
	gs.update(left=0.1, right=0.97, hspace=0)
	ax1 = plt.subplot(gs[0])
	lightcurve_plotter(filename, parser, False)
	plt.setp(ax1.get_yticklabels()[0], visible=False)    
	plt.xlabel('')
	plt.tick_params(axis='x', which='major', labelsize=0)
	ax2 = plt.subplot(gs[1])
	ax2.set_xlabel('Days', fontsize=16)
	ax2.set_ylabel('Class 1 Probability', fontsize=16, labelpad=0)
	plt.tick_params(axis='both', which='major', labelsize=16)

	survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs = parser(filename)

	print('SN type: ', sim_type)

	t = []
	prediction_a = []
	prediction_b = []
	for i in xrange(len(obs)):
		lightcurve = []
		t.append(obs[i][0])
		for j in xrange(len(obs)):
			if i >= j:
				lightcurve.append([obs[j][0], ra, decl, mwebv, hostz[0]] + obs[j][1:9])
		probability = predict_probability([lightcurve], model_str)
		print(obs[i][0], probability[-1][0], probability[-1][1])
		prediction_a.append(probability[-1][0])
		prediction_b.append(probability[-1][1])

	ax2.plot(t, prediction_a)

	plt.savefig('plots/'+filename.replace('data/SIMGEN_PUBLIC_DES/','').replace('.DAT','')+'.pdf', bbox_inches='tight', pad_inches=0.1)
	plt.show()

if __name__ == '__main__':
	'''
	Runs train()
	'''
	result = train()
