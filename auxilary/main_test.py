	
'''
Project: Neural Network for MHC Peptide Prediction
Class(s): (none) 
Function: Organizes main pipeline execution and testing 

Author: Patrick V. Holec
Date Created: 2/3/2017
Date Updated: 2/3/2017
'''

# standard libraries
import time

# nonstandard libraries
# import network     NOT READY YET
import generate_test_set as gts
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
    Sitewise Pairwise Integrated Neural Network (SPINN)
    Library: customNN 

    Mutable Parameters:
    self.test_fraction = 0.2
    self.batch_size = 100
    self.num_epochs = 500
    '''
    # network execution
    network = cnn.BuildCustomNetwork('test')

    # specifications
    network.batch_size = 250
    network.num_epochs = 300
    network.data_format()
    network.filter_initialization(form='split')
    network.network_initialization(learning_rate=0.01)
    sess = network.train()
    network.visualization(['train_accuracy']) # additional options: filters
    network.shutdown()
    
main()

