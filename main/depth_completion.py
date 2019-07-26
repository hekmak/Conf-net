# H.H Aug 2018

import os
import tensorflow as tf
from experiment import Experiment
import numpy as np
import Tkinter 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

def uncertain_loss(predictions, var, labels, weights):
    loss =  (tf.div(tf.square(predictions-labels),2*((var+1e-7)*(var+1e-7))) + (0.5)*tf.math.log((var+1e-7)*(var+1e-7)))*weights
    print "THIS IS LOSSSS"
    print loss
    return loss

# use log(var) as var
'''
def uncertain_loss(predictions, var, labels, weights):
    loss =  (tf.multiply(tf.square(predictions-labels),(0.5)*tf.math.exp(-var+1e-7)) + (0.5)*(var+1e-7))*weights
    print "THIS IS LOSSSS"
    print loss
    return loss
'''

def squared_loss(predictions, labels, weights):
    loss =  tf.square(predictions-labels)*weights
    print "THIS IS LOSSSS"
    print loss
    return loss

def load_img_to_tensor(dict_type_to_imagepath):
	dict_res = {}
	for str_type, str_filepath in dict_type_to_imagepath.items():
                if str_type == 'labelM':
			try:
			   kittipath = '/notebooks/dataset'
			   #kittipath = os.environ['KITTIPATH']
			   str_filepath = tf.regex_replace(str_filepath, tf.constant(
				'\$KITTIPATH'), tf.constant(kittipath))
			    
			except Exception:
			    print("WARNING: KITTIPATH not defined - this may result in errors!")
			tf_filepath = tf.read_file(str_filepath)
			tf_tensor = tf.image.decode_png(tf_filepath, dtype=tf.uint8)
			tf_tensor = tf.cast(tf_tensor, dtype=tf.float32)
			tf_tensor = tf.image.resize_image_with_crop_or_pad(
				tf_tensor, 352, 1216)

			dict_res[str_type] = tf_tensor
                else:
			try:
			    kittipath = '/notebooks/dataset/'
			    #kittipath = os.environ['KITTIPATH']
			    str_filepath = tf.regex_replace(str_filepath, tf.constant(
				'\$KITTIPATH'), tf.constant(kittipath))
			    
			except Exception:
			    print("WARNING: KITTIPATH not defined - this may result in errors!")
			tf_filepath = tf.read_file(str_filepath)
			tf_tensor = tf.image.decode_png(tf_filepath, dtype=tf.uint16)
			tf_tensor = tf.cast(tf_tensor, dtype=tf.int32)
			tf_tensor = tf.image.resize_image_with_crop_or_pad(
				tf_tensor, 352, 1216)

			dict_res[str_type] = tf_tensor
	return dict_res

class DepthCompletion(Experiment):

    def __init__(self):
        super(DepthCompletion, self).__init__()
        self.gpu_count = 0


    def input_fn(self, dataset, mode="train"):
        self.dict_dataset_lists = {}
        ds_input = os.path.expandvars(dataset["input"])
        ds_label = os.path.expandvars(dataset["label"])
        self.dict_dataset_lists["input"] = tf.data.TextLineDataset(ds_input)
        self.dict_dataset_lists["label"] = tf.data.TextLineDataset(ds_label)

        with tf.name_scope("Dataset_API"):
            tf_dataset = tf.data.Dataset.zip(self.dict_dataset_lists)

            if mode == "train":
                tf_dataset = tf_dataset.repeat(self.parameters.max_epochs)
                if self.parameters.shuffle:
                    tf_dataset = tf_dataset.shuffle(
                        buffer_size=self.parameters.steps_per_epoch * self.parameters.batch_size)
                tf_dataset = tf_dataset.map(load_img_to_tensor, num_parallel_calls=1)
                tf_dataset = tf_dataset.batch(self.parameters.batch_size)
                tf_dataset = tf_dataset.prefetch(buffer_size=self.parameters.prefetch_buffer_size)
            else:
                tf_dataset = tf_dataset.map(load_img_to_tensor, num_parallel_calls=1)
                tf_dataset = tf_dataset.batch(self.parameters.batch_size)#(1)
                tf_dataset = tf_dataset.prefetch(buffer_size=self.parameters.prefetch_buffer_size)

            iterator = tf_dataset.make_one_shot_iterator()

            dict_tf_input = iterator.get_next()

            tf_input = dict_tf_input["input"]
            tf_label = dict_tf_input["label"]

            '''
            tf_input = tf.map_fn(lambda img: tf.image.rgb_to_grayscale(
                img, tf_input)
            tf_label = tf.map_fn(lambda img: tf.image.rgb_to_grayscale(
                img), tf_label)
            '''

            tf_input = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, self.parameters.image_size[0], self.parameters.image_size[1]), tf_input)

            tf_label = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, self.parameters.image_size[0], self.parameters.image_size[1]), tf_label)

            tf_input.set_shape([self.parameters.batch_size,
                                self.parameters.image_size[0], self.parameters.image_size[1], 1])
            tf_input = tf.cast(tf_input, tf.float32)
            tf_label.set_shape([self.parameters.batch_size,
                                self.parameters.image_size[0], self.parameters.image_size[1], 1])
            tf_label = tf.cast(tf_label, tf.float32)
            
            # random flip
            '''
            if mode == "train":
              tf_in_lab = tf.concat([tf_input,tf_label],3)
              tf_in_lab = tf.image.random_flip_left_right(tf_in_lab)
              tf_input, tf_label = tf.split(tf_in_lab,num_or_size_splits=2,axis=3)
            '''

        tf_label = {"label":tf_label}        
        return tf_input, tf_label

    def input_fn_train(self):
        return self.input_fn( self.parameters.dataset_train )

    def input_fn_val(self):
        return self.input_fn( self.parameters.dataset_val, mode="val" )

    def input_fn_pred(self):
        return self.input_fn( self.parameters.dataset_test, mode="pred" )

    def replaceKITTIPath(self, _string):
        try:
            kittipath = os.environ['KITTIPATH']
            _string = _string.replace('$KITTIPATH', kittipath)
        except Exception:
            print("WARNING: KITTIPATH not defined - this may result in errors!")

        return _string

    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network
        pred_depth, pred_error = self.network(features, reuse=False)
        pred_depth_test, pred_error_test= self.network(features, reuse=True)
        labels = labels["label"]
      
        # for better TB visualization
        self.gpu_count = (self.gpu_count+1)%self.parameters.batch_size
        
        label_mask = tf.where(tf.equal(labels, self.parameters.invalid_value), 
                              tf.zeros_like(labels), 
                              tf.ones_like(labels))

        norm = 1. / (self.parameters.image_size[0]*self.parameters.image_size[1])

        # Predictions
        prediction = pred_depth_test
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=prediction)
            print prediction
            
        normed_label_mask = (label_mask * norm )

        # Define loss and optimizer
        self.parameters.loss_function = tf.losses.mean_squared_error# uncertain_loss#tf.losses.mean_squared_error## #
        print self.parameters.loss_function

        real_error =  tf.abs( (pred_depth-labels)*label_mask )

        real_error_constant =  tf.Variable(initial_value=tf.zeros_like(real_error),trainable=False)
        real_error_constant = real_error_constant.assign(tf.to_float(real_error))
        
        
        loss_depth = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred_depth,
                                                             labels=labels,
                                                             weights=normed_label_mask))
             
        
        loss_error = tf.reduce_mean(tf.losses.mean_squared_error( predictions=pred_error,
                                                  labels=real_error_constant,
                                                  weights=normed_label_mask))


        # normalizing losses by their value       
        Aloss_reduce1 =  tf.Variable(initial_value=0.0,trainable=False)
        new_loss1 =  tf.Variable(initial_value=0.0,trainable=False)
        Aloss_reduce2 =  tf.Variable(initial_value=0.0,trainable=False)
        new_loss2 =  tf.Variable(initial_value=0.0,trainable=False)
        
        new_loss1 = new_loss1.assign(tf.to_float(loss_depth))
        Aloss_reduce1 =  Aloss_reduce1.assign( tf.to_float( tf.abs(new_loss1) ))
        new_loss2 = new_loss2.assign(tf.to_float(loss_error))
        Aloss_reduce2 =  Aloss_reduce2.assign( tf.to_float( tf.abs(new_loss2) ))
  
        loss = loss_depth/Aloss_reduce1 + loss_error/Aloss_reduce2 

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.image("GPU"+str(self.gpu_count)+"/Input/Train/", features)
            tf.summary.image("GPU"+str(self.gpu_count)+"/Ground_truth/Train/", labels)
            tf.summary.image("GPU"+str(self.gpu_count)+"/Prediction/Dense_depth/Train/", pred_depth)
            tf.summary.image("GPU"+str(self.gpu_count)+"/Prediction/Error_map/Train/", pred_error)
            tf.summary.image("GPU"+str(self.gpu_count)+"/Prediction/Error_map_logspace/Train/", tf.log(pred_error))
            tf.summary.scalar("GPU"+str(self.gpu_count)+"/loss_depth/", loss_depth)
            tf.summary.scalar("GPU"+str(self.gpu_count)+"/loss_error/", loss_error)
            tf.summary.scalar("GPU"+str(self.gpu_count)+"/loss_depth+loss_error/", loss_depth+loss_error)

        if mode == tf.estimator.ModeKeys.EVAL:
            tf.summary.image("GPU"+str(self.gpu_count)+"/Input/Val/", features)
            tf.summary.image("GPU"+str(self.gpu_count)+"/Ground_truth/Val/", labels)

        # specify what should be done during the TRAIN call
        if mode == tf.estimator.ModeKeys.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = self.parameters.optimizer.minimize(loss=loss,
                                                              global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec( mode=mode, loss=loss, train_op=train_op)

        # specify what should be done during the EVAL call
        # Evaluate the accuracy of the model (MAE)
        mae_op = tf.metrics.mean_absolute_error(labels=labels, predictions=prediction, weights=normed_label_mask)
        # Evaluate the accuracy of the model (MSE)
        mse_op = tf.metrics.mean_squared_error(labels=labels, predictions=prediction, weights=normed_label_mask)

        return tf.estimator.EstimatorSpec( 
            mode=mode, predictions=prediction, loss=loss, eval_metric_ops={'mae': mae_op,'mse': mse_op})



    def train(self):        
        # Build the Estimator
        model = tf.estimator.Estimator(tf.contrib.estimator.replicate_model_fn(self.model_fn), model_dir=self.parameters.log_dir)

        train_spec = tf.estimator.TrainSpec(input_fn=self.input_fn_train, max_steps=self.parameters.num_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=self.input_fn_val)


        # Train and evaluate the Model
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    def val(self):
        # Clear session and prepare for testing
        pass

    def test(self):
        model = tf.estimator.Estimator(self.model_fn, model_dir=self.parameters.log_dir)
        start = time.time()
        predak = model.predict(input_fn=self.input_fn_pred)
        end = time.time()
        print "ELAPSED TIME FOR ALL: "
        print (end-start)
        i=0
        start = time.time()
        for p in predak:

             fname= os.path.join('/notebooks/project/results_haval','{:0>10d}.png'.format(i))
             fname_c= os.path.join('/notebooks/project/results_haval11','{:0>10d}.png'.format(i))
             temp = plt.imsave(arr= np.squeeze(p), fname= fname_c , cmap ='nipy_spectral')#'nipy_spectral' )
             image = np.rint(np.squeeze(p))
             image[image<900.0] = 900.0
             image = image.astype(np.uint16)
             cv2.imwrite(fname,image)
             print ('Processed image number: ',i)
             i=i+1




