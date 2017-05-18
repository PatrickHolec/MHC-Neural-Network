
# coding: utf-8

# In[32]:


'''
Project: Neural Network for MHC Peptide Prediction
Class(s): Visualize
Function: Tries to visualize some of the abilities of othe toools were building

Author: Patrick V. Holec
Date Created: 2/2/2017
Date Updated: 2/2/2017
'''

import matplotlib.pyplot as plt
import numpy as np


# makeshift checker for prediction performance
def comparison(guesses,actual):
    
    guesses = np.reshape(np.array([(guesses - min(guesses))/
                                (max(guesses)-min(guesses))]),(len(guesses),1))
    actual = np.reshape(np.array([(actual - min(actual))/
                                (max(actual)-min(actual))]),(len(actual),1)) 
    
    fig = plt.figure()
    ax = plt.gca()
    plt.scatter(guesses,actual)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Predicted Frequency')
    plt.ylabel('Actual Frequency')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show(block=False)
    raw_input('Press enter to close...')
    plt.close()
    
def auroc(guesses,actual,labels = []):

    #if type(guesses) is list:
    #    guesses = [guesses]
     
    print guesses
        
    if not labels:
        labels = ['Method {}'.format(i+1) for i in xrange(len(guesses))]
    
    guesses = [np.reshape(np.array([(guess - min(guess))/
                                (max(guess)-min(guess))]),(len(guess),1)) for guess in guesses]
    actual = np.reshape(np.array([(actual - min(actual))/
                                (max(actual)-min(actual))]),(len(actual),1)) 
    
    bin_actual = [1 if a > np.median(actual) else 0 for a in actual]
    
    lims = np.arange(1.,0.,-0.0005)
    
    x,y = [],[]
    
    for guess in guesses:
        x.append([])
        y.append([])
        for lim in lims:
            # check for set 1
            tpr,fpr = 0.,0.
            bin_guess = [1 if g > lim else 0 for g in guess]
            for g,t in zip(bin_guess,bin_actual):
                if g == 1:
                    if g == t: tpr += 1
                    else: fpr += 1
            x[-1].append(fpr/(len(bin_actual)-sum(bin_actual)))
            y[-1].append(tpr/sum(bin_actual))

    '''
    print 'Start 1:'
    for x,y in zip(x1,y1):
        print x,y
    print 'Start 2:'
    for x,y in zip(x2,y2):
        print x,y
    '''
    
    for i,j,z in zip(x,y,xrange(len(x))):
        print 'Method {} auROC: {}'.format(z+1,1.-np.trapz(i,j))    
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for i,j,l in zip(x,y,labels):
        plt.plot(i,j,label=l)
                         
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show(block=False)
    raw_input('Press enter to close...')
    plt.close()
                    
    
    
    
    