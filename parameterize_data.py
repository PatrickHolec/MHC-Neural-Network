
'''
This library is designed to parameterize existing sequence sets so we can run the default testing analysis on them
'''


# standard libraries
import pickle
import random
import os


# taking a hard datafile and doing something useful with it

class ParameterizeData:
    def __init__(self,parameters,label='test'):
        if not os.path.isfile(label+'_seqs.txt'): 
            print 'Sequence file not found. Exiting...'    
            return False
        if not all(k in parameters for k in ('length','aa_count','characters')):        
            print 'Missing one of the following keys: length, aa_count, characters. Exiting...'
            return False 
         
        # find number of sequences in selected file
        with open(label+'_seqs.txt') as fname: 
            for i,l in enumerate(fname): pass
        parameters['sequence_count'] = i
        parameters['label'] = label

        # creates a pickled file with a map of specifications
        pickle.dump(parameters,open('{}_params.p'.format(label),'wb'))
         
