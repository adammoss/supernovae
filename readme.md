## Code

Code to classify supernovae based on their lightcurves. 

## Running

First unzip the data files in the data directory by
```
tar -xvf SIMGEN_PUBLIC_DES.tar.gz
```
Next preprocess the data by 

```
python preprocess.py
```

which will create 5 random augmentations of missing data. You can train the default model (host galaxy redshift, 50% training data, unidirectional 2 layer LSTM with 16 hidden units in each layer) by

```
python run.py -f test.ini
```

After 200 epochs this should have an AUC of around 0.986, an accuracy of 94.8% and an F1 score of 0.64. The training loss should be just below the test loss. To run with a GPU (note here it is better to run with a larger batch size)

```
THEANO_FLAGS=device=gpu,floatX=float32 python run.py -f test.ini
```

To run all the combinations of models in the paper

```
python arch.py
```
