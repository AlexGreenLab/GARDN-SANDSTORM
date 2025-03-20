"""Structural Agreement calculation and visualizations

@author: aidan"""

import GA_util
import util
import nupack as n
import numpy as np

#-----------------------------------------------
def calc_struc(sequence,my_model):

    #sequence is shape 4, seq_len
    nts = GA_util.unencode(sequence)
   
    result = n.mfe(strands=[nts],model=my_model)
    return result[0].structure
#-----------------------------------------------
def return_aligned_structures(sequences):
    #Input shoudl be one-hot encoded sequences, output is an array of secondary structures for each sequence in one-hot-encoded format
    # '.' = [1,0,0]
    # '(' = [0,1,0]
    # ')' = [0,0,1]
    
    
    mod = n.Model()
    output = np.zeros((sequences.shape[0],3,sequences.shape[2]))
    for i in range(sequences.shape[0]):
        tmp_struc = str(calc_struc(sequences[i,:,:],mod))
        for j in range(len(tmp_struc)):

            if tmp_struc[j] == '.':
                output[i,0,j] = 1
                
            elif tmp_struc[j] == '(':
                output[i,1,j] = 1
                
            elif tmp_struc[j] == ')':
                output[i,2,j] = 1
    return output

#-----------------------------------------------
def calc_consensus_structure(sequences):
    
    prob = 0
    array = return_aligned_structures(sequences)
    output = np.zeros((3,sequences.shape[2]))
    
    for i in range(sequences.shape[2]):
    
        dot_sum = np.sum(array[:,0,i] == 1)
        left_parens_sum = np.sum(array[:,1,i] == 1)
        right_parens_sum = np.sum(array[:,2,i] == 1)
        
        
        output[0,i] = dot_sum / sequences.shape[0]
        output[1,i] = left_parens_sum / sequences.shape[0]
        output[2,i] = right_parens_sum / sequences.shape[0]
    
    
    return output

#-----------------------------------------------
def score_structure_against_consensus(sequences,structure):
    if sequences.shape[2] != len(structure):
        raise ValueError('Sequence and structure must be same length!')
    prob = 0
    structure_alignment = calc_consensus_structure(sequences)
    
    for i in range(len(structure)):
        if structure[i] == '.':
            prob += structure_alignment[0,i]
        elif structure[i] == '(':
            prob += structure_alignment[1,i]
        elif structure[i] == ')':
            prob += structure_alignment[2,i]
        

    return prob/ sequences.shape[2]

#-----------------------------------------------
def calc_sliding_structure_window(sequences,window_size,model=None):
    #applyt he operation on the nucleotide letters [['pandas column where sequences are held']]
    seq_len = len(sequences.iloc[0,0])

    
    output = np.zeros(shape=(sequences.shape[0],seq_len-window_size))
    for i in range(sequences.shape[0]):
        for j in range(seq_len-window_size):
            
            tmp = sequences.iloc[i,0][j:j+window_size]

            energy = n.mfe(strands=[tmp],model=model)[0].energy
            
            output[i,j] = energy
    return output


#-----------------------------------------------
def calc_defect(sequence,my_model):

    #sequence is shape 4, seq_len
    nts = GA_util.unencode(sequence)
    try:
        output_defect = n.defect(strands=[nts],structure='.'*50,model=my_model)
    
    except:
        return 1
    return output_defect

#-----------------------------------------------
def return_defects(sequences,my_model):
    #Sequences are shape nseqs,4,slen
    output = np.zeros(shape=sequences.shape[0],)
    for i in range(sequences.shape[0]):
        output[i] = calc_defect(sequences[i,:,:],my_model)
    return output

#-----------------------------------------------
def stop_codon_trace(sequences):
    
    output = np.zeros((sequences.shape[0],sequences.shape[2]//3))
    print(output.shape)
    for i in range(sequences.shape[0]):
        tmp = GA_util.unencode(sequences[i,:,:])
        # print(tmp)
        if len(tmp) == 60:
            for j in range(0,len(tmp),3):
                if tmp[j:j+3] == 'TAG' or tmp[j:j+3] == 'TAA' or tmp[j:j+3] == 'TGA':
                    output[i,j//3] = 1
        elif len(tmp) == 59:
            for j in range(2,len(tmp)-1,3):
                if tmp[j:j+3] == 'TAG' or tmp[j:j+3] == 'TAA' or tmp[j:j+3] == 'TGA':
                    output[i,j//3] = 1
            
            
    return output

#-----------------------------------------------
def return_structural_comparison_trace(sequences,structure):
    
    if sequences.shape[2] != len(structure):
        raise ValueError('Sequence and structure must be same length!')
    output = np.zeros((sequences.shape[2]))
    
    structure_alignment = calc_consensus_structure(sequences)
    
    for i in range(len(structure)):
        if structure[i] == '.':
            output[i] = structure_alignment[0,i]
        elif structure[i] == '(':
            output[i] = structure_alignment[1,i]
        elif structure[i] == ')':
            output[i] = structure_alignment[2,i]
            
    return output
