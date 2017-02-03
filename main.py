
'''
Project: Neural Network for MHC Peptide Prediction
Class(s): (none) 
Function: Organizes main pipeline execution and testing 

Author: Patrick V. Holec
Date Created: 2/3/2017
Date Updated: 2/3/2017
'''

# standard libraries

# nonstandard libraries
import network
import generate_test_set as gts

def main():
    test = gts.GenerateLandscape()
    test.define_landscape()
    test.generate_sequences()

main()

