
# coding: utf-8

# In[32]:


'''
Project: Neural Network for MHC Peptide Prediction
Class(s): BuildNetwork
Function: Gssssnerates specified neural network architecture (data agnostic)

Author: Patrick V. Holec
Date Created: 2/2/2017
Date Updated: 2/2/2017
'''


# standard libaries
import gzip
import math
import os.path
import time
import pickle
import random

# nonstandard libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import load_data as loader

# library modifications
random.seed(43)
tf.set_random_seed(43)

# main test function
def main():
    network = BuildCustomNetwork('test')
    network.data_format()
    network.filter_initialization(form='split')
    network.network_initialization(learning_rate=0.001)
    network.train()
    network.visualization(['test_accuracy']) # additional options: filters
    
# factory methods for creating variables and layers

def weight_variable(shape,custom=None):
    """Create a weight variable with appropriate initialization."""
    if custom == None: initial = tf.truncated_normal(shape, stddev=0.1)
    else: initial = tf.constant(custom,shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def constant_tensor(val,shape):
    """Create a bias variable with appropriate initialization."""
    return tf.constant(val, shape=shape,dtype=tf.float32)

def conv2d(x, W, stride=1, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def max_pool(x, stride=2, filter_size=2):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1],
                        strides=[1, stride, stride, 1], padding='SAME')

def cross_entropy(y, y_real):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_real))

def l1_loss(y,y_real):
    return tf.reduce_sum(tf.abs(tf.subtract(y,y_real)))

def l2_loss(y,y_real=None):
    if not y_real == None: return tf.nn.l2_loss(tf.subtract(y,y_real))
    else: return tf.nn.l2_loss(y)

def l2_loss_mod(y,y_real=None):
    if not y_real == None: return tf.nn.l2_loss(tf.multiply(y,tf.subtract(y,y_real)))
    else: return tf.nn.l2_loss(y)
   
def normalize_landscapes(l1,l2,dtype='sw'):
    if dtype == 'sw':
        l1_norm = l1 - np.tile(l1[0,:],(l1.shape[0],1)) 
        l2_norm = l2 - np.tile(l2[0,:],(l2.shape[0],1)) 
    return l1_norm,l2_norm

# Single fully connected layer, effectively an integrator of sitewise/pairwise elements 
def build_one_fc_layer(x_inp,Ws,bs,fn=None):
    print 'inp:',x_inp.get_shape()
    print 'Ws[0]:',Ws.get_shape()
    print 'bs[0]:',bs.get_shape()
    fc_layer = tf.add(tf.matmul(x_inp, Ws),bs)
    print 'layer:',fc_layer.get_shape()
    if fn == 'relu':
        return tf.nn.relu(fc_layer)
    elif fn == 'sigmoid':
        return tf.nn.sigmoid(fc_layer)
    else:
        return fc_layer

# Traditional build of two fully connected layer
def build_two_fc_layers(x_inp,Ws,bs,dropout=1.0,fn=None):
    print 'Generating dropout frequency of {}...'.format(dropout)
    if fn == 'relu':
        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x_inp, Ws[0]),bs[0])),dropout)
        return tf.nn.relu(tf.add(tf.matmul(h_fc1, Ws[1]),bs[1]))
    elif fn == 'sigmoid':
        h_fc1 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(x_inp, Ws[0]),bs[0])),dropout)
        return tf.nn.sigmoid(tf.add(tf.matmul(h_fc1, Ws[1]),bs[1]))
    else:
        h_fc1 = tf.nn.dropout(tf.add(tf.matmul(x_inp, Ws[0]),bs[0]),dropout)
        return tf.add(tf.matmul(h_fc1, Ws[1]),bs[1])

# Cross product of two tensors of same size
def cross_product_repeat(x_inp,Ws,dim=0):
    return tf.cross(x_inp,Ws)

'''
BuildNetwork: Main neural network architecture generator
'''

class BuildModel:
    def __init__(self,label=None,silent=False):
        print 'Initializing neural network data acquisition...'
        
        # check for label
        if not label:
            print 'No data label defined, unable to initialize any architecture.'
            return None
        
        # load data parameters
        data = loader.LoadData(label)
        self.__dict__.update(data.params)
        self.all_data_sw,self.all_data_pw = data.data_array_sw,data.data_array_pw # load all variables
        self.all_labels = data.label_array
        
        # set some basic hyperparameters
        self.test_fraction = 0.2
        self.batch_size = 100
        self.num_epochs = 500
        self.sw_depth = 1
        self.pw_depth = 1
        self.dropout = 1.0

        # params dictionary
        self.params = dict()
        self.params['fc_layers'] = 1
        self.params['hidden_units'] = 16
        
        #self.x_dim,self.y_dim = self.aa_count,self.length # maybe outdated
        self.sw_dim,self.pw_dim = self.all_data_sw.shape,self.all_data_pw.shape # save dimensions of original data
        self.full_size = (self.all_data_sw[0].size,self.all_data_pw[0].size) # number of entries in each input data
        self.pairs = (self.length*(self.length-1))/2
        
        #print 'Pairs:',self.pairs
        #print 'SW:',self.sw_dim
        #print 'PW:',self.pw_dim
 
        # verified reduction of dimension and merging
        self.flatten_data_sw = np.reshape(self.all_data_sw,
                                          (self.pw_dim[0],self.sw_dim[1]*self.sw_dim[2]))
        self.flatten_data_pw = np.reshape(self.all_data_pw,
                                          (self.pw_dim[0],self.pw_dim[1]*self.pw_dim[2]*self.pw_dim[3]))
        self.all_data = np.concatenate((self.flatten_data_sw,self.flatten_data_pw),axis=1)
         
        # update on model parameters
        if not silent:
            print '*** System Parameters ***'
            print '  - Sequence length:',self.length
            print '  - AA count:',self.aa_count
        
        print 'Finished acquisition!'

        # Create GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Start engine
        print 'Initializing variables...'
        self.sess = tf.Session(config=config)
    
    def data_format(self,silent=False,augment=False,normalization=False,**kwargs):
        print 'Starting data formatting...'
        
        # randomize data order
        order,limit = range(0,self.all_data.shape[0]),int((1-self.test_fraction)*self.all_data.shape[0])
        np.array(random.shuffle(order))

        # normalize label energy
        if normalization == True:
            self.all_labels = np.reshape(np.array([(self.all_labels - min(self.all_labels))/
                                                  (max(self.all_labels)-min(self.all_labels))]),(len(self.all_labels),1))
        
        # alternative normalization
        self.all_labels = np.reshape(np.array(self.all_labels),(len(self.all_labels),1))
        
        # split data into training and testing
        self.train_data = self.all_data[np.array(order[:limit]),:]
        self.test_data = self.all_data[np.array(order[limit:]),:]
        self.train_labels = self.all_labels[np.array(order[:limit]),:]
        self.test_labels = self.all_labels[np.array(order[limit:]),:]
         
        # data augmentation
        if augment == True:
            
            self.train_labels,self.test_labels = self.train_labels - np.min(self.all_labels),self.test_labels - np.min(self.all_labels)
 
            train_total = sum([int((10**l[0])**0.5) for l in self.train_labels])
            test_total = sum([int((10**l[0])**0.5) for l in self.test_labels]) 

            self.train_data_augment,self.train_labels_augment = np.zeros((train_total,self.full_size[0] + self.full_size[1])),np.zeros((train_total,1))
            self.test_data_augment,self.test_labels_augment = np.zeros((test_total,self.full_size[0] + self.full_size[1])),np.zeros((test_total,1))

            ind1,ind2 = 0,0
    
            for d,l in zip(self.train_data,self.train_labels):
                for i in xrange(int((10**l)**0.5)): 
                    self.train_data_augment[ind1,:] = d
                    self.train_labels_augment[ind1,:] = l
                    ind1 += 1

            for d,l in zip(self.test_data,self.test_labels):
                for i in xrange(int((10**l)**0.5)): 
                    self.test_data_augment[ind2,:] = d
                    self.test_labels_augment[ind2,:] = l
                    ind2 += 1
             
            print 'Train data (augment):',self.train_data_augment.shape
            print 'Test data (augment)::',self.test_data_augment.shape
            print 'Train labels (augment):',self.train_labels_augment.shape
            print 'Train labels (augment):',self.test_labels_augment.shape
            
            self.train_data,self.test_data = self.train_data_augment,self.test_data_augment
            self.train_labels,self.test_labels = self.train_labels_augment,self.test_labels_augment

        print 'Train data:',self.train_data.shape
        print 'Test data:',self.test_data.shape
        print 'Train labels:',self.train_labels.shape
        print 'Train labels:',self.test_labels.shape

        # normalize label energy
        #self.train_labels =  np.array([(self.train_labels - min(self.all_labels))/(max(self.all_labels)-min(self.all_labels))])#,self.train_labels.size,1)
        #self.test_labels =  np.array([(self.test_labels - min(self.all_labels))/(max(self.all_labels)-min(self.all_labels))])#,self.test_labels.size,1)

        print 'Finished formatting!'

    def filter_initialization(self,form='variable',sw_depth=None,pw_depth=None):
        # possible last minute modifications
        if not sw_depth is None: self.sw_depth = sw_depth
        if not pw_depth is None: self.pw_depth = pw_depth

        # still working on sw/pw depth for filters
        print 'Initializing filters...'
        data = pickle.load(open('test_map.p'))
        sw_landscape = data['sw_landscape']
        if self.params['fc_layers'] == 1:
            if form == 'variable':
                self.W_sw = weight_variable((self.aa_count,self.length,1,self.sw_depth))
                self.W_pw = weight_variable((self.aa_count,self.aa_count,1,self.pairs))
                self.W_fc = weight_variable((self.length + self.pairs,1))        
                self.b_fc = bias_variable((1,1))
            elif form == 'constant':
                self.W_sw = constant_tensor([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]],(self.aa_count,self.length,1,1))
                self.W_pw = constant_tensor(0,(self.aa_count,self.aa_count,1,self.pairs))
                self.W_fc = constant_tensor(1,(self.length + self.pairs,1))        
                self.b_fc = constant_tensor(0,(1,1))
            elif form == 'split':
                self.W_sw = weight_variable((self.aa_count,self.length,1,self.sw_depth))
                self.W_pw = weight_variable((self.aa_count,self.aa_count,1,self.pairs))
                #self.W_pw = constant_tensor(0,(self.aa_count,self.aa_count,1,self.pairs))
                self.W_fc = constant_tensor(1,(self.length + self.pairs,1))        
                #self.b_fc = constant_tensor(0,(1,1))
                self.b_fc = bias_variable((1,1))
            elif form == 'custom':
                self.W_sw = weight_variable((self.aa_count,self.length,1,1),custom=np.transpose(sw_landscape))
                self.W_pw = constant_tensor(0,(self.aa_count,self.aa_count,1,self.pairs))
                self.W_fc = constant_tensor(1,(self.length + self.pairs,1))        
                self.b_fc = constant_tensor(0,(1,1))
        
        elif self.params['fc_layers'] == 2:
            hidden_units = self.params['hidden_units']
            if True:
                self.W_sw = weight_variable((self.aa_count,self.length,1,self.sw_depth))
                self.W_pw = weight_variable((self.aa_count,self.aa_count,1,self.pairs))
                self.W_fc = weight_variable([self.length + self.pairs,hidden_units])        
                self.b_fc = bias_variable([hidden_units])
                self.W_fc2 = weight_variable([hidden_units,1])        
                self.b_fc2 = bias_variable([1])
    ''' 
    Provides output class variables:
    self.train_x,self.train_y
    '''

    def network_initialization(self,layers=2,learning_rate=0.01,beta=0.1,fn=None):

        # Load data 
        self.train_x = tf.placeholder(tf.float32, shape=(None, sum(self.full_size))) # full vector input
        self.train_y = tf.placeholder(tf.float32, shape=(None, 1)) # full energy input

        # Split the pw/sw entries into two streams
        train_x_sw,train_x_pw = tf.split(self.train_x,[self.full_size[0],self.full_size[1]],1)

        #x_image_sw = tf.transpose(tf.reshape(train_x_sw, [-1, self.aa_count, self.length, 1]))
        x_image_sw = tf.reshape(train_x_sw, [-1, self.aa_count, self.length, 1])
        x_image_pw = tf.reshape(train_x_pw, [-1, self.pairs, self.aa_count, self.aa_count])        
        x_image_pw = tf.transpose(x_image_pw, [0, 2, 3, 1])

               # create image/histogram summaries

        # visualizations for filters
        tf.summary.image('W_sw',tf.reshape(self.W_sw,(1,self.aa_count,self.length,1)))
        #tf.summary.image('W_pw',self.W_pw)
        #tf.summary.histogram('W_sw',self.W_sw)
        #tf.summary.histogram('W_pw',self.W_pw)

        # creating sitewise convolution
        with tf.name_scope('sitewise_layer'):
            conv_sw_array = [conv2d(a,b,stride=1,padding='VALID') 
                        for a,b in zip(tf.split(x_image_sw,[1 for i in xrange(self.length)],2),
                                       tf.split(self.W_sw,[1 for i in xrange(self.length)],1))]  
            conv_sw = tf.concat(conv_sw_array,1)   

        # creating pairwise convolution
        with tf.name_scope('pairwise_layer'):
            conv_pw_array = [conv2d(a,b,stride=1,padding='VALID') 
                        for a,b in zip(tf.split(x_image_pw,[1 for i in xrange(self.pairs)],3),
                                       tf.split(self.W_pw,[1 for i in xrange(self.pairs)],3))]  
            conv_pw = tf.concat(conv_pw_array,1)   

        # concatenate sw/pw contributions, fully connected
        with tf.name_scope('fc_layer'):
            full_conv = tf.concat([conv_sw,conv_pw],1)
            full_conv_flat = tf.squeeze(full_conv)

        if self.params['fc_layers'] == 1: 
            print 'Creating one fully connected layer...'
            self.y_conv = build_one_fc_layer(full_conv_flat,self.W_fc,self.b_fc,fn=fn)
        elif self.params['fc_layers'] == 2: 
            print 'Creating two fully connected layers...'
            self.y_conv = build_two_fc_layers(full_conv_flat,[self.W_fc,self.W_fc2],[self.b_fc,self.b_fc2],dropout=self.dropout,fn=fn) 

        # model access variables
        with tf.name_scope('loss'): 
            self.loss = l2_loss(self.y_conv,self.train_y) + beta*l2_loss(self.W_pw) + beta*l2_loss(self.W_sw)
            #self.loss = l2_loss_mod(self.y_conv,self.train_y) + beta*l2_loss(self.W_pw) + beta*l2_loss(self.W_sw)
            tf.summary.scalar('loss',self.loss)

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
       
    def train(self,silent=True):
        # start timer
        start = time.time()
        
        # training hyperparameters

        # training via iterative epochs
        batches_per_epoch = int(len(self.train_data)/self.batch_size)
        num_steps = int(self.num_epochs * batches_per_epoch)
        
        print 'Batchs per epoch - {} / Number of steps - {}'.format(batches_per_epoch,num_steps)
        
        init = tf.global_variables_initializer()
        
        self.sess.run(init)

        writer = tf.summary.FileWriter('./logs/nn_logs',self.sess.graph)
        merged = tf.summary.merge_all()
        
        epoch_loss = 0
        epoch_acc = 0

        if not silent:
            plt.axis([-0.2,0.1,-0.2,0.1])
            plt.ion()

        for step in xrange(num_steps):
            offset = (step * self.batch_size) % (self.train_data.shape[0] - self.batch_size)

            batch_x = self.train_data[offset:(offset + self.batch_size), :]
            batch_y = np.reshape(self.train_labels[offset:(offset + self.batch_size)],(self.batch_size,1))

            feed_dict = {self.train_x: batch_x, self.train_y: batch_y}
            
            _, batch_loss = self.sess.run([self.train_step, self.loss],feed_dict=feed_dict)
            #print 'Batch loss:',batch_loss 
            epoch_loss += batch_loss
                        
            if (step % batches_per_epoch == 0):

                epoch_loss /= batches_per_epoch*self.batch_size

                feed_dict = {self.train_x: self.test_data, self.train_y: self.test_labels}
                batch_loss_validation = self.sess.run(self.loss,feed_dict=feed_dict)

                print 'Avg batch loss at step %d: %f' % (step, epoch_loss)
                print 'Batch loss ({})  /  Validation loss ({})'.format(batch_loss,batch_loss_validation)

                epoch_loss = 0
                # randomize input data
                together = np.concatenate((self.train_data,self.train_labels),axis=1)
                np.random.shuffle(together)
                self.train_data = together[:,:-1]
                self.train_labels = np.reshape(together[:,-1],(self.train_labels.shape[0],1)) # need to add dimension to data
                
                # summary
                summary = self.sess.run(merged,feed_dict=feed_dict)
                writer.add_summary(summary,step)    
                
                # interactive plot 
                sw_predict = np.reshape(self.sess.run(self.W_sw),(self.aa_count,self.length))
                data = pickle.load(open('test_map.p'))
                sw_landscape = np.transpose(data['sw_landscape'])
                sw_predict,sw_landscape = normalize_landscapes(sw_predict,sw_landscape) # normalize for shift factors 
                
                if not silent: self.plot_landscape(sw_predict,sw_landscape,pause=0.05)

        if not silent: self.plot_landscape(sw_predict,sw_landscape,pause=50)


        #print 'Sitewise filter:\n',sw_predict
        #print 'True landscape:\n',sw_landscape
        print 'Training time: ', time.time() - start
        

    def plot_landscape(self,swp,swl,pause=1):
        plt.cla()
        plt.scatter(np.reshape(swp,(swp.size,1)),np.reshape(swl,(swl.size,1)))
        plt.pause(pause)
                
    def visualization(self,picks=['test_accuracy','filters'],suspend = 5.0):
        # start engine
        
        if 'train_accuracy' in picks:
            # visualize
            predicted_labels = self.sess.run(self.y_conv, feed_dict={self.train_x: self.train_data})
            # fitting
            fig,ax = plt.subplots()
            predicted_labels,real_labels = np.squeeze(predicted_labels),np.squeeze(self.train_labels)
            fit = np.polyfit(np.array(predicted_labels),real_labels,deg=1)
            ax.plot(predicted_labels,fit[0]*predicted_labels + fit[1], color='green')

            # data plotting
            ax.scatter(predicted_labels,real_labels)
            plt.show(block=False)

            if suspend == True:
                raw_input('Press enter to close.')
            else:
                plt.pause(suspend)
            plt.close()

        if 'test_accuracy' in picks:
            # visualize
            predicted_labels = self.sess.run(self.y_conv, feed_dict={self.train_x: self.test_data})
            # fitting
            fig,ax = plt.subplots()
            predicted_labels,real_labels = np.squeeze(predicted_labels),np.squeeze(self.test_labels)
            fit = np.polyfit(np.array(predicted_labels),real_labels,deg=1)
            ax.plot(predicted_labels,fit[0]*predicted_labels + fit[1], color='green')

            # data plotting
            ax.scatter(predicted_labels,real_labels)
            plt.show(block=False)

            if suspend == True:
                raw_input('Press enter to close.')
            else:
                plt.pause(suspend)
            plt.close()
                 
        if 'filters' in picks:
            # layer 1 weights
            feed_dict={self.train_x: self.test_data}
            A = self.sess.run(self.W_sw,feed_dict=feed_dict)
            print 'A:',A
            sh = A.shape
            A = np.reshape(A,(sh[0],sh[1]))
            if True:
                plt.imshow(A, cmap='jet', interpolation='nearest')
                plt.title('Filter (Layer 1)')
                plt.xlabel('AA Index')

            plt.show(block=False)
            if suspend == True:
                raw_input('Press enter to close.')
            else:
                plt.pause(suspend)
            plt.close()

        if 'auroc' in picks:
            predicted_labels = self.sess.run(self.y_conv, feed_dict={self.train_x: self.test_data})
            # fitting
            score,y= np.squeeze(predicted_labels),np.squeeze(np.round(self.test_labels))

            score = (score - min(score))/(max(score) - min(score))

            roc_x = []
            roc_y = []
            min_score = min(score)
            max_score = max(score)
            thr = np.linspace(min_score, max_score, 30)
            FP=0
            TP=0
            N = sum(y)
            P = len(y) - N

            for (i, T) in enumerate(thr):
                for i in range(0, len(score)):
                    if (score[i] > T):
                        if (y[i]==1):
                            TP = TP + 1
                        if (y[i]==0):
                            FP = FP + 1
                roc_x.append(FP/float(N))
                roc_y.append(TP/float(P))
                FP=0
                TP=0

            print 'auROC score:',roc_auc_score(y,score)

            plt.scatter(roc_x, roc_y)
            plt.show(block=False)

            if suspend == True:
                raw_input('Press enter to close.')
            else:
                plt.pause(suspend)
            plt.close()


    def parameter_export(self):
        feed_dict = {self.train_x: batch_x, self.train_y: batch_y}
        self.sess(W_sw,feed_dict=feed_dict)

    def shutdown(self):
        self.sess.close()        
        
# namespace activation
if __name__ == '__main__':
    main()







