

'''
This project was started to try and determine how well we can implement regularization in our curent model
This particular script generates test landscapes and sequences at various sample sizes
Intended to be used with a downstream reconstruction script
'''

# standard libraries


# nonstandard libraries
import numpy as np

# configuration


# main

def main():
    # main hypothesis testing
    pass

# helper functions


# generation class

class GenerateLandscape:
    def __init__(self):
        self.length = 5
        self.aa_count = 5
        self.sw_mag = 3 # sitewise distribution magnitude
        self.pw_mag = 1 # pairwise distribution magnitude
        self.noise_mag = 0.1 # sequence generation distribution magnitude
        self.sequence_count = 1000 
        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def define_landscape(self):
        self.sw_landscape = np.random.normal(0,self.sw_mag,(self.length,self.aa_count))        
        self.pw_landscape = np.random.normal(0,self.pw_mag,(self.length-1,self.length-1,self.aa_count**2))        

    def generate_sequences(self,count=None, filename='test_seqs.txt'):

        # check for viability
        self.check_params()

        # Quick access to the number of sequences you want
        if count:
            self.sequence_count = count

        # create sequence list
        self.sequences = []
        raw_seqs = np.random.randint(0,self.aa_count,(self.sequence_count,self.length))
        self.sequences = [''.join([self.characters[i] for i in row]) for row in raw_seqs]
        self.scores = [sum([self.sw_landscape[ind][i] for ind,i in enumerate(row)]+
            [self.pw_landscape[ind1][ind2-1][self.aa_count*i + j] for ind1,i in enumerate(row[:-1]) for ind2,j in enumerate(row[ind1:])]) for row in raw_seqs]

        # write file
        with open(filename,'w') as myfile:
            for seq,score in zip(self.sequences,self.scores):
                myfile.write('{},{}\n'.format(seq,score))

    # check for parameter validity prior to code execution
    def check_params(self):
        assert self.length <= len(self.characters),'Not enough characters to represent amino acid repetoire.'
     

# namespace identifier

if __name__ == '__main__':
    main()

# closing notes



