
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

# personal libraries
from models import * # libraries: kNN,cNN,spiNN
from analysis import * # libraries: visualize
from landscape import * # libraries: generate_test_set,load_data,parameterize_data

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
    test.unique_sequence_count = 2000 
    test.sw_mag = 1 # sitewise distribution magnitude
    test.pw_mag = 0.5 # pairwise distribution magnitude
    test.noise_mag = 0.5 # sequence generation distribution magnitude
    test.length = 7
    test.aa_count = 5

    # main
    test.define_landscape()
    test.generate_sequences()
    test.generate_records()

    time.sleep(2.0)
    

    '''
    Data Parameterization
    Note: what comes out of the landscape generator is a sequence followed by an enrichment value
    Library: parameterize_data 

    Mutable Parameters:
    '''
    #parameters = {'length':14,'aa_count':21,'characters':'ACDEFGHIKLMNPQRSTVWY_'}
    #pd.ParameterizeData(parameters,label='A12')
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
    
    ### TESTING CODE ###
    # options: test,A12_01,A12_12,A12_23,A12_34
    set_name = 'A12' # name of the data we are working with
    test_fraction = 0.25
    models = ['knn','spinn']
    
    all_guesses = []
    
    if 'knn' in models:
        ## kNN Model
        print 'Starting kNN model...'

        data_settings = {
                     'num_epochs':100,'learning_rate':0.001,
                     'data_augment':False,'data_normalization':True,
                     'fc_fn':('linear','linear'),
                     'test_fraction': test_fraction,
                     'silent':False
                    }


        model = kNN.BuildModel(set_name,data_settings)
        model.data_format()
        guesses1 = model.predict()
        all_guesses.append(guesses1)
        #vis.comparison(guesses1,model.test_labels)
    
    if 'cnn' in models:
        ## cNN Model
        print 'Starting cNN model...'

        data_settings = {
                     'num_epochs':100,'learning_rate':0.001,
                     'data_augment':False,'data_normalization':True,
                     'data_silent':True,
                     'test_fraction': test_fraction
                    }


        model = cNN.BuildModel(set_name,data_settings)
        model.data_format()
        model.network_initialization()
        model.train()
        guesses2 = model.predict()
        all_guesses.append(guesses2)
        #vis.comparison(guesses2,model.test_labels)

    if 'fc' in models:
        ## FC Model
        print 'Starting FC model...'

        data_settings = {
                     'num_epochs':100,'learning_rate':0.001,
                     'data_augment':False,'data_normalization':True,
                     'data_silent':True,
                     'test_fraction': test_fraction,
                     # number of layers
                     'cnn_layers':0, # includes conv & pool
                     'fc_layers':3,
                     # fully connected parameters
                     'fc_depth':(32,16,1),
                     'fc_fn':('relu','relu','linear'),
                     'fc_dropout':(0.5,0.5,1.0)
                    }


        model = cNN.BuildModel(set_name,data_settings)
        model.data_format()
        model.network_initialization()
        model.train()
        guesses3 = model.predict()
        all_guesses.append(guesses3)
        #vis.comparison(guesses3,model.test_labels)

    if 'spinn' in models:
        ## spiNN Model
        print 'Starting spiNN model...'

        data_settings = {
                         'num_epochs':30,'learning_rate':0.001,
                         'data_augment':False,'data_normalization':True,
                         'test_fraction': test_fraction,
                         'fc_fn':('linear','linear'),
                         'fc_layers':1,
                         'fc_depth':(1,1),
                         'fc_dropout':(1.0,1.0),
                         'silent':False
                        }


        model = spiNN.BuildModel(set_name,data_settings)
        model.data_format()
        model.network_initialization()
        model.train()
        guesses4 = model.predict()
        all_guesses.append(guesses4)
        #vis.comparison(guesses4,model.test_labels)

    
    vis.auroc(all_guesses,
               model.test_labels,labels = models)
    
    
    
    
    ## OLD MODEL EXECUTION, want to make sure I didn't forget to recycle anything here
    
    '''
    # network execution
    network = spinn.BuildModel('test')

    # specifications
    network.test_fraction = 0.9
    network.dropout = 1.0 
    network.batch_size = 20
    network.num_epochs = 100 
    network.sw_layers = 1
    network.params['fc_layers'] = 2
    network.params['hidden_units'] = 16
    network.data_format(augment=False,normalization=True)
    network.filter_initialization(form='variable')
    network.network_initialization(learning_rate=0.005,fn='sigmoid',beta=0.01)
    sess = network.train()
    network.visualization(['auroc'],suspend = True)
    network.visualization(['test_accuracy'],suspend = True) # additional options: filters
    #network.visualization(['train_accuracy'])
    network.shutdown()
    '''
    
main()

