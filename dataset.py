from __future__ import print_function
import numpy as np
import csv

sn1a_classifier = {'1':0, '2':1, '3':1, '21':1, '22': 1, '23':1, '32': 1, '33':1}
type_classifier = {'1':0, '2':1, '3':2, '21':1, '22': 1, '23':1, '32': 2, '33':2}
subtype_classifier = {'1':0, '21':1, '22':2, '23':3, '32': 4, '33':5, '3':6}

def pad_sequences(sequences, maxlen=None, dtype='int32',
				  padding='pre', truncating='pre', value=-1.):
	'''
	Pads each sequence to the same length, which is the length of the longest sequence. Returns
        a numpy array of the padded sequences. 
	* maxlen is an integer which truncates any sequence longer than maxlen and fills
          any sequence less than maxlen up to the value of maxlen
	* dtype is a string containing the type to cast the resulting sequence
	* padding is a string, either "pre" or "post" which dictates whether to pad at the start
	  or end of the sequence
	* truncating is a string, either "pre" or "post" which dictates whether to truncate at 
          the start or end of the sequence
	* value is a float containing the value to pad the sequences with
	* x is the numpy array containing the padded sequences with with dimensions 
	  (number_of_sequences, maxlen)
	- Used in load_data() when padding the training and test data
	- Used in predict_probability() in train.py to pad the light curve data
	'''
	lengths = [len(s) for s in sequences]

	nb_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max(lengths)

	# take the sample shape from the first non empty sequence
	# checking for consistency in the main loop below.
	sample_shape = tuple()
	for s in sequences:
		if len(s) > 0:
			sample_shape = np.asarray(s).shape[1:]
			break

	x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
	for idx, s in enumerate(sequences):
		if len(s) == 0:
			continue  # empty list was found
		if truncating == 'pre':
			trunc = s[-maxlen:]
		elif truncating == 'post':
			trunc = s[:maxlen]
		else:
			raise ValueError('Truncating type "%s" not understood' % truncating)

		# check `trunc` has expected shape
		trunc = np.asarray(trunc, dtype=dtype)
		if trunc.shape[1:] != sample_shape:
			raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
							 (trunc.shape[1:], idx, sample_shape))

		if padding == 'post':
			x[idx, :len(trunc)] = trunc
		elif padding == 'pre':
			x[idx, -len(trunc):] = trunc
		else:
			raise ValueError('Padding type "%s" not understood' % padding)
	return x

def to_categorical(y, nb_classes=None):
	'''
	Convert class vector (integers from 0 to nb_classes) to binary class matrix,
        for use with categorical_crossentropy.
	* y is an list of labels 
	* nb_classes contains each of the unique classes
	* Y is a binary class matrix
	- Used in load_data()
	'''
	if not nb_classes:
		nb_classes = np.max(y)+1
	Y = np.zeros((len(y), nb_classes))
	for i in range(len(y)):
		Y[i, y[i]] = 1.
	return Y

def load_data(path="data/unblind_nohostz", classifier=sn1a_classifier, test_fraction=0.2, nb_augment=1, seed=None):
	'''
	Loads data from the files produced by preprocess.py. Returns the data as numpy arrays which can be used in the
        model.
	* path is a string containing the path and base name of the preprocessed data
	* classifier is a dictionary containing which classification is wanted (sn1a_classifier for SN1a vs. non-SN1a, 
          type_classifier for SN1 vs. SN2 vs. SN3 or subtype_classifier for SN1a vs SN1b vs ...)
	* test_fraction is a float between 0 and 1 describing the fraction of the data to be used to test the network,
          1-test_fraction is the fraction of the data used to train the network
        * nb_augment is the number of random augmentations of data to use
	* seed is an integer required to prevent different random values
	* nb_samples is the number of different supernovae
	* sequence_length is the length of the sequences in each supernova event (all equal using pad_sequence())
	* output_dim is the number of elements in the results (12 with no host and 13 with host)
	* length_train is the number of different supernovae used to train the model
	* length_test is the number of different supernovae used to test the model 
	* X_train contains the data to train the model and has dimensions (length_train, sequence_length, output_dim)
	* X_train_reverse contains the data to train the model and has dimensions (length_train, sequence_length, output_dim)
	  and is the same as X_train but the padding is applied to the opposite end and then the sequence reversed.
	  This is used for bidirectional models
	* Y_train contains the binary class matrix of the training labels formed using to_categorical() and is a 1 tensor 
          of shape (length_train)
	* ids_train is a list of the indices of the data to be used in the training sample
	* X_test contains the data to test the model and has dimensions (length_test, sequence_length, output_dim)
	* X_test_reverse contains the data to test the model and has dimensions (length_test, sequence_length, output_dim)
	  and is the same as X_train but the padding is applied to the opposite end and then the sequence reversed.
	  This is used for bidirectional models
	* Y_test contains the binary class matrix of the test labels formed using to_categorical() and is a 1 tensor 
          of shape (length_test)
	* ids_test is a list of the indices of the data to be used in the test sample
	- Used in train() in train.py to load the data
	! Uses pad_sequences() to create the correct length sequences
	! Uses to_categorical() to create the binary class matrix
	'''

	last_id = None
	ids = []

	np.random.seed(seed)
	
	with open(path+'_1.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			id = int(row[0])
			if id != last_id:
				ids.append(id)
				last_id = id

	ids = np.array(ids)
	length = ids.shape[0]
	test_length = int(length*test_fraction)
	indices = np.random.permutation(length)
	training_idx, test_idx = indices[:length-test_length], indices[length-test_length:]
	ids_train = ids[training_idx]
	ids_test = ids[test_idx]
	
	labels = []
	data_sequence = []
	data = []
	training_idx = []
	test_idx = []
	idx = 0

	ids_train_permute = []
	ids_test_permute = []

	for i in range(1, nb_augment+1):

		print('Reading dataset: ',path+'_'+str(i)+'.csv')

		last_id = None
		first_time = True

		with open(path+'_'+str(i)+'.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				id = int(row[0])
				if id != last_id:
					if not first_time:
						data.append(data_sequence)
						if id in ids_train:
							training_idx.append(idx)
							ids_train_permute.append(last_id)
						else:
							test_idx.append(idx)
							ids_test_permute.append(last_id)
						labels.append(last_label)
						idx += 1
						data_sequence = []
					first_time = False
				last_id = id
				last_label = classifier[row[-1]]
				inputs = [float(v) for i, v in enumerate(row[1:-2])]			
				data_sequence.append(inputs)

	data_copy = list(data)
	data = pad_sequences(data, dtype='float')
	data_post = pad_sequences(data_copy, dtype='float', padding='post')

	labels = np.array(labels)

	training_idx = np.array(training_idx)
	test_idx = np.array(test_idx)
	
	length = data.shape[0]
	sequence_len = data.shape[1]
	output_dim = data.shape[2]

	nb_classes = np.unique(labels).shape[0]

	X_train = data[training_idx,:,:]
	X_train_reverse = data_post[training_idx,::-1,:]
	X_test = data[test_idx,:,:]
	X_test_reverse = data_post[test_idx,::-1,:]
	Y_train = labels[training_idx]
	Y_test = labels[test_idx]
	
	length_train = training_idx.shape[0]
	length_test = test_idx.shape[0]

	Y_train = to_categorical(Y_train, nb_classes)
	Y_test = to_categorical(Y_test, nb_classes)

	ids_train = np.array(ids_train_permute)
	ids_test = np.array(ids_test_permute)
	
	return (X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), (length_train, length_test, sequence_len, output_dim, nb_classes)
