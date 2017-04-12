
'''
This file is for primarily testing basic TensorFlow setups
'''

import random
import numpy as np

aa_count,length = 5,3
uniques = 100

raw_seqs = random.sample(range(0,125),uniques)
raw_seqs_2 = [[(a/(aa_count**l))%aa_count for l in xrange(length)] for a in raw_seqs]
raw_seqs_2 = [''.join(map(str,s)) for s in raw_seqs_2]
print len(raw_seqs_2)
print len(set(raw_seqs_2))
