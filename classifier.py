from __future__ import print_function
from IPython.display import Image

import os
import numpy as np
import matplotlib.pyplot as plt
import math

from cntk.layers import default_options, Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense, Sequential, For
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms 
from cntk.initializer import glorot_uniform, he_normal
from cntk import Trainer
from cntk.learner import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error, relu, input_variable, softmax, element_times
from cntk.utils import *



#def func_X(z):
#	return softmax(z)


#if __name__ == '__main__'
def create_basic_model(input, out_dims):
	    
    net = Convolution((5,5), 32, init=glorot_uniform(), activation=relu, pad=True)(input)
    net = MaxPooling((3,3), strides=(2,2))(net)

    net = Convolution((5,5), 32, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3,3), strides=(2,2))(net)

    net = Convolution((5,5), 64, init=glorot_uniform(), activation=relu, pad=True)(net)
    net = MaxPooling((3,3), strides=(2,2))(net)
    
    net = Dense(64, init=glorot_uniform())(net)
    net = Dense(out_dims, init=glorot_uniform(), activation=None)(net)
    
    return net


# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10

#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("This tutorials depends 201A tutorials, please run 201A first.")

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8) # train uses data augmentation (translation only)
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))



#
# Train and evaluate the network.
#
class train_evaluate:
	def __init__(self,max_epochs = 1,epoch_size = 50000,minibatch_size = 16):
		self.maxepochs = max_epochs
		self.epochsize = epoch_size
		self.minibatchsize = minibatch_size
	def train_and_evaluate(self,reader_train, reader_test, max_epochs, model_func):
	    # Input variables denoting the features and label data
	    input_var = input_variable((num_channels, image_height, image_width))
	    label_var = input_variable((num_classes))

	    # Normalize the input
	    feature_scale = 1.0 / 256.0
	    input_var_norm = element_times(feature_scale, input_var)
	    

	    # apply model to input
	    self.z = model_func(input_var_norm, out_dims=10)

	    #
	    # Training action
	    #

	    # loss and metric
	    ce = cross_entropy_with_softmax(self.z, label_var)
	    pe = classification_error(self.z, label_var)

	    # training config
	    # epoch_size     = 50000
	    # minibatch_size = 64

	    # Set training parameters
	    lr_per_minibatch       = learning_rate_schedule([0.01]*10 + [0.003]*10 + [0.001], UnitType.minibatch, self.epochsize)
	    momentum_time_constant = momentum_as_time_constant_schedule(-self.minibatchsize/np.log(0.9))
	    l2_reg_weight          = 0.001
	    
	    # trainer object
	    learner     = momentum_sgd(self.z.parameters, 
	                               lr = lr_per_minibatch, momentum = momentum_time_constant, 
	                               l2_regularization_weight=l2_reg_weight)
	    trainer     = Trainer(self.z, (ce, pe), [learner])

	    # define mapping from reader streams to network inputs
	    input_map = {
	        input_var: reader_train.streams.features,
	        label_var: reader_train.streams.labels
	    }

	    log_number_of_parameters(self.z) ; print()
	    progress_printer = ProgressPrinter(tag='Training', num_epochs=self.maxepochs)

	    # perform model training
	    batch_index = 0
	    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
	    for epoch in range(self.maxepochs):       # loop over epochs
	        sample_count = 0
	        while sample_count < self.epochsize:  # loop over minibatches in the epoch
	            data = reader_train.next_minibatch(min(self.minibatchsize, self.epochsize - sample_count), input_map=input_map) # fetch minibatch.
	            trainer.train_minibatch(data)                                   # update model with it

	            sample_count += data[label_var].num_samples                     # count samples processed so far
	            
	            # For visualization...            
	            plot_data['batchindex'].append(batch_index)
	            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
	            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)
	            
	            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
	            batch_index += 1
	        progress_printer.epoch_summary(with_metric=True)
	    
	  

	    # process minibatches and evaluate the model
	    metric_numer    = 0
	    metric_denom    = 0
	    sample_count    = 0
	    minibatch_index = 0

	    while sample_count < self.epochsize:
	        current_minibatch = min(self.minibatchsize, self.epochsize - sample_count)

	        # Fetch next test min batch.
	        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

	        # minibatch data to be trained with
	        metric_numer += trainer.test_minibatch(data) * current_minibatch
	        metric_denom += current_minibatch

	        # Keep track of the number of samples processed so far.
	        sample_count += data[label_var].num_samples
	        minibatch_index += 1

	    print("")
	    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
	    print("")
	    
	    # Visualize training result:
	    window_width            = 32
	    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
	    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

	    # Moving average.
	    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
	    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
	    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
	    
	    plt.figure(1)
	    plt.subplot(211)
	    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
	    plt.xlabel('Minibatch number')
	    plt.ylabel('Loss')
	    plt.title('Minibatch run vs. Training loss ')

	    plt.show()

	    plt.subplot(212)
	    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
	    plt.xlabel('Minibatch number')
	    plt.ylabel('Label Prediction Error')
	    plt.title('Minibatch run vs. Label Prediction Error ')
	    plt.show()
	    return softmax(self.z)

	    #z=func_X(z)
	def call_me(self):
	    return softmax(self.z)
	def train(self):
		data_path = os.path.join('data', 'CIFAR-10')
		reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
		reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)


		pred = self.train_and_evaluate(reader_train, reader_test, max_epochs=1, model_func=create_basic_model)	   
		a=self.call_me()
	   


#print(pred)


#c_obj = train_evaluate()


