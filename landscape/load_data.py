
import pickle
import numpy as np

class LoadData:
    def __init__(self,label='test',scope=['sw','pw']):

        # hold variable names (in case undefined)
        self.data_array_sw,self.data_array_pw,self.label_array = None,None,None
        
        # open file and store lines
        with open('{}_seqs.txt'.format(label),'r') as f:
            content = f.readlines()
        
        # pull out parameters from the pickeled params file
        self.params = pickle.load(open('{}_params.p'.format(label),'rb'))
        
        # split up lines it data and labels
        self.raw_data,self.raw_labels = [],[]
        for data in [' '.join(c.split()).split(',') for c in content]:
            self.raw_data.append(data[0])
            self.raw_labels.append(float(data[1]))
        
        # always make label array
        self.label_array = np.reshape(np.array(self.raw_labels),(len(self.raw_labels),1,1))
            
        # one-hot encoding (sitewise)
        if 'sw' in scope:
            self.data_array_sw = np.zeros((len(self.raw_data),self.params['aa_count'],self.params['length']),np.int)
            for i,sample in enumerate(self.raw_data):
                for j,char in enumerate(sample):
                    try: self.data_array_sw[i,self.params['characters'].index(char),j] = 1
                    except ValueError:
                        print self.params['characters'], char
                        raw_input()
                
        # one-hot encoding (pairwise)
        if 'pw' in scope:
            pair_count = (self.params['length']*(self.params['length']-1))/2
            self.data_array_pw = np.zeros((len(self.raw_data),pair_count,self.params['aa_count'],self.params['aa_count']))
            for i,sample in enumerate(self.raw_data):
                pair_index = 0
                for j,char1 in enumerate(sample):
                    for k,char2 in enumerate(sample[j+1:]):
                        self.data_array_pw[i,pair_index,self.params['characters'].index(char1),
                                                        self.params['characters'].index(char2)] = 1
                        pair_index += 1
