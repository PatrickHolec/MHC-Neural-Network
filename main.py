
'''
Project: Neural Network for MHC Peptide Prediction
Class(s): (none) 
Function: Organizes main pipeline execution and testing 

Author: Patrick V. Holec
Date Created: 2/3/2017
Date Updated: 3/20/2017

This is for actual data testing

'''

# standard libraries
import time

# nonstandard libraries
import generate_test_set as gts
import parameterize_data as pd
import customNN as cnn

def main():
    '''
    Landscape Generator
    Library: generate_test_set 

    Mutable Parameters:
    self.length = 5
    self.aa_count = 6
    self.sw_mag = 1 # sitewise distribution magnitude
    self.pw_mag = 0 # pairwise distribution magnitude
    self.noise_mag = 0.000 # sequence generation distribution magnitude
    self.sequence_count = 2000 
    '''
    '''
    # initialization
    test = gts.GenerateLandscape()

    # specifications
    test.unique_sequence_count = 500 
    test.sw_mag = 1 # sitewise distribution magnitude
    test.pw_mag = 0.5 # pairwise distribution magnitude
    test.noise_mag = 1.5 # sequence generation distribution magnitude
    test.length = 7
    test.aa_count = 5

    # main
    test.define_landscape()
    test.generate_sequences()
    test.generate_records()

    time.sleep(2.0)
    '''

    '''
    Data Parameterization
    Library: parameterize_data 

    Mutable Parameters:
    '''
    parameters = {'length':14,'aa_count':21,'characters':'ACDEFGHIKLMNPQRSTVWY_'}
    pd.ParameterizeData(parameters,label='A12')
    '''
    Sitewise Pairwise Integrated Neural Network (SPINN)
    Library: customNN 

    Mutable Parameters:
    self.test_fraction = 0.2
    self.batch_size = 100
    self.num_epochs = 500

    Functions:
        def __init__(self,label=None,silent=False):
        def data_format(self,silent=False,augment=False,normalization=False,**kwargs):
        def network_initialization(self,layers=2,learning_rate=0.01,beta=0.1,fn=None):

    '''
    # network execution
    network = cnn.BuildCustomNetwork('A12')

    # specifications
    network.batch_size = 20
    network.num_epochs = 50 
    network.params['fc_layers'] = 1
    network.data_format(augment=False,normalization=True)
    network.filter_initialization(form='variable')
    network.network_initialization(learning_rate=0.001,fn='sigmoid',beta=0.01)
    sess = network.train()
    network.visualization(['test_accuracy']) # additional options: filters
    network.visualization(['train_accuracy'])
    network.shutdown()
    
main()

