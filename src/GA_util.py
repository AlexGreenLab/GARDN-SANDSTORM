"""
Created on Tue Mar 29 20:28:48 2022

@author: aidan
"""

import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
import tensorflow.keras
import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import Conv2D
import copy
import nupack as n

#-----------------------------------------------
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

EPSILON = 1e-20





nucleotides = {'A':0,'G':1,'C':2,'T':3,'a':0,'g':1,'c':2,'t':3}
letters = ['A','G','C','T']
bp_lookup = {
'G_C' :[1, 0, 0, 0, 0, 0, 0],
'C_G' :[0, 1, 0, 0, 0, 0, 0],
'A_T' :[0, 0, 1, 0, 0, 0, 0],
'T_A' :[0, 0, 0, 1, 0, 0, 0],
'G_T' :[0, 0, 0, 0, 1, 0, 0],
'T_G' :[0, 0, 0, 0, 0, 1, 0],
'N_C' :[0, 0, 0, 0, 0, 0, 1]
}


#Sequence stacks are shape (nseqs, 4, slen) in NN pipeline
#-----------------------------------------------
def get_random_nt():
    val = np.random.uniform()
    if val <= 0.25:
        return 'A'
    elif 0.25 < val <= 0.5:
        return 'G' 
    elif 0.5 < val <= 0.75:
        return 'C'
    elif 0.75 < val <=1:
        return 'T'
    
#-----------------------------------------------
def get_diff_nt(nt):
    if nt == 'A':
        return np.random.choice(['G','C','T'])
    elif nt == 'G':
        return np.random.choice(['A','C','T'])
    elif nt == 'C':
        return np.random.choice(['A','G','T'])
    elif nt == 'T':
        return np.random.choice(['A','G','C'])
    
#-----------------------------------------------   
def unencode_nt(nt):
    if np.array_equal(nt, np.array([1,0,0,0])):
        return 'A'
    elif np.array_equal(nt, np.array([0,1,0,0])):
        return 'G'
    elif np.array_equal(nt, np.array([0,0,1,0])):
        return 'C'
    elif np.array_equal(nt, np.array([0,0,0,1])):
        return 'T'
    
#-----------------------------------------------
def unencode(sequence):
    out = ''
    for i in range(sequence.shape[1]):
        val = np.argmax(sequence[:,i])
        out += letters[val]
    return out  


#-----------------------------------------------
def take_random_sequence(seqs):
    choice = np.random.choice(0,seqs.shape[0])
    return seqs[choice,:,:]

#-----------------------------------------------
#create a single random sequence
def create_rand_seq(slen):
    #creates seq of shape (4,slen)
    out = np.zeros(shape = (4,slen))
    for i in range(slen):
        nt = get_random_nt()
        out[nucleotides[nt],i] = 1
    return out

#-----------------------------------------------
#Create multiple random sequences
def create_mult_rand_seqs(nseq,slen):
    #returns seqs of shape (nseq,4,slen)
    out = np.zeros(shape=(nseq,4,slen))
    for i in range(nseq):
        out[i,:,:] = create_rand_seq(slen)
    return out

#-----------------------------------------------
#Mutate a specific sequence with a given frequency of mutation
def mutate_seq(seq,freq):
    #Seq is shape (4,slen)
    
    #number of nts to mutate
    mutate_num = round(seq.shape[1] * freq)
    
    #Select random nts to change
    #This sometimes returns multiple of the same choice
    choices = np.random.choice(seq.shape[1],mutate_num)
    
    
    #Mutate the sequence
    tmp = [unencode_nt(seq[:,choices][:,i]) for i in range(len(choices))]
    for i in range(len(tmp)):
        seq[:,choices[i]] = 0
        seq[nucleotides[get_diff_nt(tmp[i])],choices[i]] = 1 
    
    return seq

#-----------------------------------------------
#mutate multiple sequences
def mutate_mult_seqs(seqs,freq):
    #seqs is shape (nseqs,4,slen)
    for i in range(seqs.shape[0]):
        seqs[i,:,:] = mutate_seq(seqs[i,:,:],freq)
    return seqs

#-----------------------------------------------
#Cross over the contents of two sequences at n_points different locations
def cross_over(seq_1,seq_2,n_points):
    #seq_1 and seq_2 are shape (4,slen)
    out = copy.deepcopy(seq_1)
    
    
    choices = np.random.choice(seq_1.shape[1],n_points,replace=False)
    choices.sort()

    for i in range(len(choices)-1):
        if i == 0:
            out[:,:choices[i]] = seq_2[:,:choices[i]]
            
        elif i == len(choices) - 1:
            out[:,choices[i]:] = seq_2[:,choices[i]:]
            
        else:
            out[:,choices[i]:choices[i+1]] = seq_2[:,choices[i]:choices[i+1]]
            
    return out
    
#-----------------------------------------------
#Crossover random combinations of members of seqs    
def cross_over_mult_seqs(seqs,n_returned,n_points=2):
    
    #n_returned is how many sequences will actually be returned, seqs is the pool of parents that will be crossed over 
    out = np.zeros(shape=(n_returned,seqs.shape[1],seqs.shape[2]))
    for i in range(n_returned):
        idx = np.random.choice(seqs.shape[0],2,replace=False)
        seq_1 = seqs[idx[0],:,:]
        seq_2 = seqs[idx[1],:,:]
        out[i,:,:] = cross_over(seq_1,seq_2,n_points)
    return out


#-----------------------------------------------
def revcomp(s):
    #s is shape (4,slen)
    a = letters.index('A')
    c = letters.index('C')
    g = letters.index('G')
    t = letters.index('T')
    
    sr= s.copy()
    sr[a,:]=s[t,:]
    sr[t,:]=s[a,:]
    
    sr[c,:]=s[g,:]
    sr[g,:]=s[c,:]

    sr=np.fliplr(sr)
    
    
    return sr    

#-----------------------------------------------
def revcompstr(seq):
    # print('input must not contain "U" bases')
    try:
        seq = seq.replace('U','T')
    except:
        pass
    seq = seq.replace("A", "t").replace("C", "g").replace("T", "a").replace("G", "c")
    seq = seq.upper()
    seq = seq[::-1]
    return seq




#-----------------------------------------------
def revcompmult(s):
    #s is shape(n,4,slen)
    a = letters.index('A')
    c = letters.index('C')
    g = letters.index('G')
    t = letters.index('T')
    
    sr=s.copy()
    sr[:,a,:]=s[:,t,:]
    sr[:,t,:]=s[:,a,:]
    
    sr[:,c,:]=s[:,g,:]
    sr[:,g,:]=s[:,c,:]

    sr=np.flip(sr,axis=2)
    
    
    return sr

    
#-----------------------------------------------   
def calc_GC(sequences):
    #sequences is shape(n,4,slen)
    a = letters.index('A')
    t = letters.index('T')
    s = sequences.copy()
    s[:,a,:] = 0
    s[:,t,:] = 0
    
    gc_content = np.sum(s,axis=(2,1))
    gc_content = gc_content / sequences.shape[2]
    
    #returns a list of the fraction gc content that each sequence contains
    return gc_content

#-----------------------------------------------

def is_complement(nt1,nt2):
    #Boolean output
    #This function lives in one hot encoding space and was designed because np arrays cant be compared
    #a return t
    if np.argmax(nt1) == 0 and np.argmax(nt2) == 3:
        return 1
    #g return c
    elif np.argmax(nt1) == 1 and np.argmax(nt2)==2:
        return 1
    #c return g
    elif np.argmax(nt1) == 2 and np.argmax(nt2) == 1:
        return 1
        
    #t return a
    elif np.argmax(nt1) == 3 and np.argmax(nt2)==0:
        return 1
    else:
        return 0 
    


#-----------------------------------------------
def pairwise_prob_fastest(seq):
    
    #computes the structure array
    
    #seq is shape 4, nt
    seq_len = seq.shape[1]
    
    output = np.zeros(shape = (seq_len,seq_len))
    
    #Find the indices of each nucleotide
    As = np.where(np.argmax(seq,axis=0)==0)
    Gs = np.where(np.argmax(seq,axis=0)==1)
    Cs  = np.where(np.argmax(seq,axis=0)==2)
    Ts = np.where(np.argmax(seq,axis=0)==3)
    
    
    #Make it so now every G-C pair is represented as a 3
    output[Gs,:] = 1
    output[:,Cs]+=2
    output[output<3] = 0
    
    output[Cs,:] = 1
    output[:,Gs] +=2
    output[output<3] = 0
    
    #Repeat for every A-T base pair
    output[As,:] = 1
    output[:,Ts] += 1
    
    output[output< 2] = 0
    

    output[Ts,:] = 1
    output[:,As] +=1
    output[output<2] = 0
    

    
    #Now we need to divide by the distance between the nucleotides
    where_vals_gc = np.where(output==3)
    where_vals_at = np.where(output==2)
    
    #dividing base pairs by distance without for loops
    distance_vec_gc = np.abs(where_vals_gc[0] - where_vals_gc[1])
    # output[where_vals_gc] =1 
    output[where_vals_gc] /= distance_vec_gc
    
    distance_vec_at = np.abs(where_vals_at[0] - where_vals_at[1])
    output[where_vals_at] /= distance_vec_at
    
    row_sums = output.sum(axis=1)
    new_matrix = output / row_sums[:, np.newaxis]
    #Nan catch for columns with no binding partners
    new_matrix = np.nan_to_num(new_matrix)

    return new_matrix


#-----------------------------------------------
def create_ppms_fast(seqs):
    if seqs.shape[0] is None:
        return None
    else:
        output = np.zeros(shape=(seqs.shape[0],seqs.shape[2],seqs.shape[2]))
        for i in range(seqs.shape[0]):
            output[i,:,:] = pairwise_prob_fastest(seqs[i,:,:])
        return output
    
    
#-----------------------------------------------
def contact_map(seq):
    
    #computes the structure array
    
    #seq is shape 4, nt
    seq_len = seq.shape[1]
    
    output = np.zeros(shape = (seq_len,seq_len)) #increasing dimension to 2 for G-U pair calculation
    
    gu_output = np.zeros(shape = (seq_len,seq_len))
    
    #Find the indices of each nucleotide
    As = np.where(np.argmax(seq,axis=0)==0)
    Gs = np.where(np.argmax(seq,axis=0)==1)
    Cs  = np.where(np.argmax(seq,axis=0)==2)
    Ts = np.where(np.argmax(seq,axis=0)==3)
    
    
    #Make it so now every G-C pair is represented as a 3
    output[Gs,:] = 1
    output[:,Cs]+=2
    output[output<3] = 0
    
    output[Cs,:] = 1
    output[:,Gs] +=2
    output[output<3] = 0
    
    
    #Repeat for every A-U base pair
    output[As,:] = 1
    output[:,Ts] += 1
    
    output[output< 2] = 0
    

    output[Ts,:] = 1
    output[:,As] +=1
    output[output<2] = 0
    
    #Repeat for every G-U base pair
    gu_output[Gs,:] = 1
    gu_output[:,Ts] += 1
    
    gu_output[gu_output< 2] = 0
    

    gu_output[Ts,:] = 1
    gu_output[:,Gs] +=1
    gu_output[gu_output<2] = 0
    

    
#     #Now we need to divide by the distance between the nucleotides
#     where_vals_gc = np.where(output==3)
#     where_vals_at = np.where(output==2)
    
#     #dividing base pairs by distance without for loops
#     distance_vec_gc = np.abs(where_vals_gc[0] - where_vals_gc[1])
#     # output[where_vals_gc] =1 
#     output[where_vals_gc] /= distance_vec_gc
    
#     distance_vec_at = np.abs(where_vals_at[0] - where_vals_at[1])
#     output[where_vals_at] /= distance_vec_at
    
    
#     where_vals_gu = np.where(gu_output==2)
#     distance_vec_gu = np.abs(where_vals_gu[0] - where_vals_gu[1])
#     gu_output[where_vals_gu] /= distance_vec_gu


    
    # return output,gu_output
    
    output +=gu_output
    
#     # row_sums = output.sum(axis=1)
#     # new_matrix = output / row_sums[:, np.newaxis]
#     # #Nan catch for columns with no binding partners
#     # new_matrix = np.nan_to_num(new_matrix)

    return output # Just do the binarization


#-----------------------------------------------
def prototype_ppms_fast(seqs):
    if seqs.shape[0] is None:
        return None
    else:
        output = np.zeros(shape=(seqs.shape[0],seqs.shape[2],seqs.shape[2]))
        for i in range(seqs.shape[0]):
            output[i,:,:] = contact_map(seqs[i,:,:])
        return output


#-----------------------------------------------

def return_nupack_ppm_fast(sequence,my_model):
    nts = unencode(sequence)
    # my_model = n.Model()

    probability_matrix = n.pairs(strands=[nts], model=my_model)
    return probability_matrix.to_array()



#-----------------------------------------------
def create_ppms_nupack(sequences,my_model):
    #sequences is shape nseqs,4,slen

    return np.array([return_nupack_ppm_fast(x,my_model) for x in sequences])

#-----------------------------------------------
def return_vismap(sequence):
    #Sequence is shape 4, sequence_length
    
    
    #Axis 0 and 1 are positions in the array, axis 3 is the one hot encoding label describing the base pair
    output = np.zeros(shape = (sequence.shape[1],sequence.shape[1]))
    
    #Using these lookup table prevents the need for slow pyhton for loops
    As = np.where(np.argmax(sequence,axis=0)==0)
    Gs = np.where(np.argmax(sequence,axis=0)==1)
    Cs  = np.where(np.argmax(sequence,axis=0)==2)
    Ts = np.where(np.argmax(sequence,axis=0)==3)
    
    #Access the indices of gc bases
    output[Gs,:] = 1
    output[:,Cs]+=7
    output[output<8] = 0
    
    #Save the base pair indices
    where_vals_gc = np.where(output==8)
    
    #Now accesss the indices of cg bases
    output[Cs,:] = 1
    output[:,Gs]+=6
    output[output<7] = 0
    
    where_vals_cg = np.where(output==7)
    
    #AT indices
    output[As,:] = 1
    output[:,Ts] += 5
    output[output<6] = 0
    
    where_vals_at = np.where(output==6)
    
    #TA indices
    output[Ts,:] = 1
    output[:,As] += 4
    output[output<5] = 0
    
    where_vals_ta = np.where(output==5)


    
    #Non Canonical base pairs is the rest
    where_vals_nc = np.where(output<3)
    

    output[where_vals_gc[0],where_vals_gc[1]] = 4
    output[where_vals_cg[0],where_vals_cg[1]] = 3
    output[where_vals_at[0],where_vals_at[1]] = 2
    output[where_vals_ta[0],where_vals_ta[1]] = 1
    # output[where_vals_gt[0],where_vals_gt[1]] = 1
    # output[where_vals_tg[0],where_vals_tg[1]] = 1
    output[where_vals_nc[0],where_vals_nc[1]] = -1
    return output

#-----------------------------------------------
def create_vismaps(sequences):
    #sequences is shape n_seqs,4,seq_len
    output = np.zeros(shape=(sequences.shape[0],sequences.shape[2],sequences.shape[2]))
    for i in range(sequences.shape[0]):
        output[i,:,:] = return_vismap(sequences[i,:,:])
    return output



#-----------------------------------------------
def calc_struc(sequence,my_model):

    #sequence is shape 4, seq_len
    nts = unencode(sequence)
   
    result = n.mfe(strands=[nts],model=my_model)
    return result[0].structure

#-----------------------------------------------
def calc_mfe(sequence,my_model):

    #sequence is shape 4, seq_len
    nts = unencode(sequence)
   
    result = n.mfe(strands=[nts],model=my_model)
    return result[0].energy
#-----------------------------------------------
def calc_mfe_str(sequence,my_model):

    #sequences are strings   
    result = n.mfe(strands=[sequence],model=my_model)
    return result[0].energy

#-----------------------------------------------
def return_mfes(sequences,my_model):
    #Sequences are shape nseqs,4,slen
    output = np.zeros(shape=sequences.shape[0],)
    for i in range(sequences.shape[0]):
        output[i] = calc_mfe(sequences[i,:,:],my_model)
    return output
#-----------------------------------------------
def calc_defect(sequence,my_model):

    #sequence is shape 4, seq_len
    nts = unencode(sequence)
    if len(nts) == 59:
        try:
            output_defect = n.defect(strands=[nts],structure='............(((((((((...((((((...........))))))...)))))))))',model=my_model)
    
        except:
            return 1
    else:
        try:
            output_defect = n.defect(strands=[nts],structure='.............(((((((((...((((((...........))))))...)))))))))',model=my_model)
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
def calc_struc_dist(sequence,my_model):
    #sequence is shape 4, slen
    nts = unencode(sequence)
	
    if len(nts) == 59:
        return n.struc_distance('............(((((((((...((((((...........))))))...)))))))))',calc_struc(sequence,my_model))
    else:
        return n.struc_distance('.............(((((((((...((((((...........))))))...)))))))))',calc_struc(sequence,my_model))


#-----------------------------------------------
def return_struc_distances(sequences,my_model):
    #Sequences is shape nseqs,4,slen
    output = np.zeros(shape=sequences.shape[0],)
    for i in range(sequences.shape[0]):
        output[i] = calc_struc_dist(sequences[i,:,:],my_model)
    return output

#-----------------------------------------------
def plot_model_structure_quality(generator_model,n_switches=1000,latent_dim=100,my_model=None):
    #inputs should be a trained generator model and the nubmer of switches to generate to evaluate 2nd structure
    
    #This is how you get the latent dim from a trained model
    # try:
    #     latent_dim = generator_model.layers[0].output_shape[0][1]
   
    
    
    random_latent_vectors = tf.random.normal(shape=(n_switches, latent_dim))
    
    # Decode them to fake switches

    generated_switches = generator_model(random_latent_vectors)
    
    

    dist_save = []
    for i in range(n_switches):
        dist_save.append(calc_struc_dist(generated_switches[i,:,:,0],my_model))
    print('Mean:',np.mean(dist_save))
    print('Variance:',np.var(dist_save))
    plt.figure()
    plt.title('Generator Structure Distance')
    plt.hist(dist_save,density='True',alpha=0.7,color='firebrick')
    plt.xlim([0,60])
    plt.subplot().spines['right'].set_visible(False)
    plt.subplot().spines['top'].set_visible(False)
    plt.legend(['GAN Output'])
    plt.subplot().spines['left'].set_linewidth(2.0)
    plt.subplot().spines['bottom'].set_linewidth(2.0)
    plt.subplot().xaxis.set_tick_params(width=1.5)
    plt.subplot().yaxis.set_tick_params(width=1.5)
    plt.show()
    
    return generated_switches,dist_save


#-----------------------------------------------
def decode_vismap(vismap):
    #vismap is shape s_len, s_len
    #Has to be an encoding scheme where AT is different than TA and GC is different than CG
    output = ''
    dictionary = {4:'C',3:'G',2:'T',1:'A'}
    
   
    #iterate over the columns
    for i in range(vismap.shape[0]):
        idx = np.where(vismap[:,i] != -1)[0][0]
        output += dictionary[vismap[idx,i]]
    return output


#-----------------------------------------------
DNA_Codons = {
    # 'M' - START, '_' - STOP
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TGT": "C", "TGC": "C",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TTT": "F", "TTC": "F",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H",
    "ATA": "I", "ATT": "I", "ATC": "I",
    "AAA": "K", "AAG": "K",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATG": "M",
    "AAT": "N", "AAC": "N",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TGG": "W",
    "TAT": "Y", "TAC": "Y",
    "TAA": "_", "TAG": "_", "TGA": "_"
}



#-----------------------------------------------
def gumbel_distribution(input_shape: Tuple[int, ...]) -> tf.Tensor:
    """Samples a tensor from a Gumbel distribution.
    Args:
        input_shape: Shape of tensor to be sampled.
    Returns:
        (tf.Tensor): An input_shape tensor sampled from a Gumbel distribution.
    """

    # Samples an uniform distribution based on the input shape
    
    uniform_dist = tf.random.uniform(input_shape, 0, 1)

    # Samples from the Gumbel distribution
    gumbel_dist = -1 * tf.math.log(
        -1 * tf.math.log(uniform_dist + EPSILON) + EPSILON
    )

    return gumbel_dist

#-----------------------------------------------
class GumbelSoftmax(Layer):
    """A GumbelSoftmax class is the one in charge of a Gumbel-Softmax layer implementation.
    References:
        E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax.
        Preprint arXiv:1611.01144 (2016).
    """

    def __init__(self, axis: Optional[int] = -1, **kwargs) -> None:
        """Initialization method.
        Args:
            axis: Axis to perform the softmax operation.
        """

        super(GumbelSoftmax, self,tau).__init__(**kwargs)

        # Defining a property for holding the intended axis
        self.axis = axis
        self.tau = tau

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.
        Args:
            x: A tensorflow's tensor holding input data.
            tau: Gumbel-Softmax temperature parameter.
        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): Gumbel-Softmax output and its argmax token.
        """

        # Adds a sampled Gumbel distribution to the input
        x = inputs + gumbel_distribution(tf.shape(inputs))

        # Applying the softmax over the Gumbel-based input
        x = tf.nn.softmax(x * self.tau, self.axis)

        # Sampling an argmax token from the Gumbel-based input
        # y = tf.stop_gradient(tf.argmax(x, self.axis, tf.int32))

        return x

    def get_config(self) -> Dict[str, Any]:
        """Gets the configuration of the layer for further serialization.
        Returns:
            (Dict[str, Any]): Configuration dictionary.
        """

        config = {"axis": self.axis}
        base_config = super(GumbelSoftmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
	
#-----------------------------------------------	
# with tf.device('device:GPU:0'):
class SelfAttention(tf.keras.layers.Layer):
    
    def __init__(self,kernel=(1,1),stride=(1,1)):
        super(SelfAttention, self).__init__()
        self.kernel = kernel
        self.stride = stride
        

    def build(self, input_shape):
        
        self.batch_size,self.height,self.width,self.channels = input_shape

        
        
    def attention_calc(self, inputs):
    
        f = Conv2D(filters=self.channels // 8, kernel_size=self.kernel, strides=self.stride, name='f_conv')(inputs)  # [bs, h, w, c']
        # f = max_pooling(f)
       
        g = Conv2D(filters=self.channels // 8, kernel_size=self.kernel, strides=self.stride, name='g_conv')(inputs)  # [bs, h, w, c']
        
        h = Conv2D(self.channels // 2, kernel_size=self.kernel, strides=self.stride, name='h_conv')(inputs)  # [bs, h, w, c]
        


        # N = h * w
        f_g_reshape = tf.keras.layers.Reshape(target_shape=(inputs.shape[1]*inputs.shape[2],inputs.shape[3]//8))
        h_reshape = tf.keras.layers.Reshape(target_shape=(inputs.shape[1]*inputs.shape[2],inputs.shape[3]//2))
        

        s = tf.matmul(f_g_reshape(g), f_g_reshape(f), transpose_b=True)  # # [bs, N, N]
        

        beta = tf.nn.softmax(s,axis=-1)
        # attention map

        o = tf.matmul(beta, h_reshape(h))
        
        # [bs, N, C]
        gamma = tf.Variable(initial_value=0.,dtype=tf.float32,trainable=True)
    
        # o = tf.reshape(o, shape=[self.batch_size, self.height, self.width, self.channels // 2])  # [bs, h, w, C]
        
        o = tf.keras.layers.Reshape(target_shape=[self.height,self.width,self.channels//2])(o)
        o = Conv2D(filters=self.channels, kernel_size=self.kernel, strides=self.stride,name='attention_convolution')(o)
        
        inputs = (gamma * o) + inputs

        return inputs

    def call(self, inputs):
        return self.attention_calc(inputs)
#-----------------------------------------------	
RBS = np.array([[1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1.],
                [0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
AUG = np.array([[1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 0.],
       [0., 1., 0.,]])

C = np.array([[0.],
 [0.],
 [1.],
 [0.]])	
#-----------------------------------------------


#-----------------------------------------------
#Creates data with an RBS in the correct position
def create_rbs_data(n_seqs):
    left = create_mult_rand_seqs(n_seqs,30)
    RBS_stack = np.stack([RBS for i in range(n_seqs)])
    right = create_mult_rand_seqs(n_seqs,18)
    output = np.concatenate((left,RBS_stack,right),axis=2)
    return output

#-----------------------------------------------
#Creates Data that has a start codon in the correct position
def create_aug_data(n_seqs):
    left = create_mult_rand_seqs(n_seqs,47)
    AUG_stack = np.stack([AUG for i in range(n_seqs)])
    right = create_mult_rand_seqs(n_seqs,9)
    output = np.concatenate((left,AUG_stack,right),axis=2)
    return output

#-----------------------------------------------
#Creates data with both an RBS and Start Codon, but does not necessarily fold correctly
def create_rbs_aug_data(n_seqs):
    switch = create_mult_rand_seqs(n_seqs,30)
    RBS_stack = np.stack([RBS for i in range(n_seqs)])
    stem_1 = create_mult_rand_seqs(n_seqs,6)
    AUG_stack = np.stack([AUG for i in range(n_seqs)])
    stem_2 = create_mult_rand_seqs(n_seqs,9)
    output = np.concatenate((switch,RBS_stack,stem_1,AUG_stack,stem_2),axis=2)
    return output

#-----------------------------------------------   
#Creates data that is reverse complement of itself
def create_binding_data(n_seqs):
    x = create_mult_rand_seqs(n_seqs,60)
    
    FIVE_PRIME = x[:,:,:12]
    STEM_1 = x[:,:,12:30]
    RC = revcompmult(STEM_1)
    print(RC.shape)
    
    x = np.concatenate((FIVE_PRIME,STEM_1,x[:,:,30:42],RC[:,:,:6],x[:,:,48:51],RC[:,:,9:]),axis=2)
    x = x[:,:,1:]
    return x
#-----------------------------------------------  
def create_C_stack(n):
    C = np.array([[0.],
 [0.],
 [1.],
 [0.]])	

    C_stack = np.stack([C for _ in range(n)])
    return C_stack

#-----------------------------------------------
def create_valeri_model():
    input_seqs = keras.Input(shape=(59,4))

    a = layers.Conv1D(10, (5,), strides=(1,), padding="same",activation='linear')(input_seqs)
    a = layers.Conv1D(5, (3,), strides=(1,), padding="same",activation='linear')(a)
    a = layers.Flatten()(a)
    a = layers.Dropout(0.1)(a)

    a = layers.Dense(150,activation='relu')(a)
    a = layers.Dropout(0.1)(a)

    a   = layers.Dense(60,activation='relu')(a)
    a = layers.Dropout(0.1)(a)

    a = layers.Dense(15,activation='relu')(a)

    on_output = layers.Dense(1,activation='linear',name='on_output')(a)
    off_output = layers.Dense(1,activation='linear',name='off_output')(a)




    valeri_model  = keras.Model(inputs=input_seqs,outputs=[on_output,off_output],name='valeri_model')
    return valeri_model

#-----------------------------------------------
def create_joint_model(seq_len=60,ppm_len=60,latent_dim=128,internal_activation='relu',output_activation='linear',output_nodes=1,output_units=1):
    # seq_len is the length of the sequences
    # Latent dim controls the number of filters
    # internal activation is a tensorflow activation function, applied to the hiden layers
    # output activation is the activation function of the model output
    # output nodes is the number of different output channels the model will have (e.g. 1 for UTR prediciton, 2 for Toehold Prediction (on, off)
    # output units is the number of units in the output layer, 1 for regression problems and N for classification with N different classes 
    


    #Define the Inputs
    input_seqs = keras.Input(shape=(4,seq_len,1))
    input_probs = keras.Input(shape=(ppm_len,ppm_len,1))


    # Predictive Model Definition
    #Sequence Branch
    y = tfkl.Conv2D(latent_dim/4, (4, 18), strides=(4, 1), padding="same",activation=internal_activation)(input_seqs)
    y = tfkl.BatchNormalization()(y)
    # y = layers.SpatialDropout2D(0.2)(y)
    y = tfkl.Conv2D(latent_dim/8, (4, 9), strides=(4, 1), padding="same",activation=internal_activation)(y)
    y = tfkl.Conv2D(latent_dim/16,(4,3),strides=(4,1),padding='same',activation=internal_activation)(y)
    y = tfkl.Flatten()(y)

    #PPM Branch    
    x = tfkl.Conv2D(latent_dim/4, (8, 8), strides=(2, 2), padding="same",activation=internal_activation)(input_probs)
    # x = layers.SpatialDropout2D(0.2)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Conv2D(latent_dim/8, (4, 4), strides=(2, 2), padding="same",activation=internal_activation)(x)
    x = tfkl.Conv2D(latent_dim/16,(2,2),strides=(2,2),padding='same',activation=internal_activation)(x)
    x = tfkl.Flatten()(x)



    #Combine the two
    z = tfkl.Concatenate()([x,y])
    z = tfkl.Dense(16,activation=internal_activation)(z)
    z = tfkl.Dense(8,activation=internal_activation)(z)
    z = tfkl.Dense(4,activation=internal_activation)(z)
	
    
    output_lst = []
    for i in range(output_nodes):
        tmp = tfkl.Dense(output_units,activation=output_activation,name='prediction_output_%s'%i)(z)
        output_lst.append(tmp)


    output_model  = keras.Model(inputs=[input_seqs,input_probs],outputs=output_lst,name='joint_model')
    return output_model

#-----------------------------------------------
def create_SANDSTORM(seq_len=60,ppm_len=60,latent_dim=128,internal_activation='relu',output_activation='linear',output_nodes=1,output_units=1,kernel_1_size=[4,18],kernel_2_size=[4,9],kernel_3_size=[4,3]):
    # seq_len is the length of the sequences
    # Latent dim controls the number of filters
    # internal activation is a tensorflow activation function, applied to the hiden layers
    # output activation is the activation function of the model output
    # output nodes is the number of different output channels the model will have (e.g. 1 for UTR prediciton, 2 for Toehold Prediction (on, off)
    # output units is the number of units in the output layer, 1 for regression problems and N for classification with N different classes 
    


    #Define the Inputs
    input_seqs = keras.Input(shape=(4,seq_len,1))
    input_probs = keras.Input(shape=(ppm_len,ppm_len,1))


    # Predictive Model Definition
    #Sequence Branch
    y = layers.Conv2D(latent_dim/4, kernel_1_size, strides=(4, 1), padding="same",activation=internal_activation)(input_seqs)
    y = layers.BatchNormalization()(y)
    # y = layers.SpatialDropout2D(0.2)(y)
    y = layers.Conv2D(latent_dim/8, kernel_2_size, strides=(4, 1), padding="same",activation=internal_activation)(y)
    # y = layers.BatchNormalization()(y)
    y = layers.Conv2D(latent_dim/16,kernel_3_size,strides=(4,1),padding='same',activation=internal_activation)(y)
    # y = layers.BatchNormalization()(y)
    y = layers.Flatten()(y)

    #PPM Branch    
    x = layers.Conv2D(latent_dim/4, (9,9), strides=(ppm_len, 1), padding="same",activation=internal_activation)(input_probs)
    # x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.Conv2D(latent_dim/8, (5,5), strides=(ppm_len, 1), padding="same",activation=internal_activation)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Conv2D(latent_dim/16, (3,3),strides=(ppm_len,1),padding='same',activation=internal_activation)(x)
    x = layers.GlobalMaxPooling2D()(x)


    x = layers.Flatten()(x)

    # z = CrossAttentionLayer()([x,y])


    #Combine the two
    z = layers.Concatenate()([x,y])
    z = layers.Dense(16,activation=internal_activation)(z)
    z = layers.Dense(8,activation=internal_activation)(z)
    z = layers.Dense(4,activation=internal_activation)(z)
	
    
    output_lst = []
    for i in range(output_nodes):
        tmp = layers.Dense(output_units,activation=output_activation,name='prediction_output_%s'%i)(z)
        output_lst.append(tmp)


    output_model  = keras.Model(inputs=[input_seqs,input_probs],outputs=output_lst,name='joint_model')
    return output_model


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
#-----------------------------------------------#-----------------------------------------------
def calc_consensus_structure(sequences):
    #Returns the maximum likelihood structure and the probability of that structure calculated by forward algorithm
    
    prob = 1
    array = return_aligned_structures(sequences)
    print(array.shape)
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
    
    prob = 0
    structure_alignment = calc_consensus_structure(sequences)
    
    for i in range(len(structure)):
        if structure[i] == '.':
            prob += structure_alignment[0,i]
        elif structure[i] == '(':
            prob += structure_alignment[1,i]
        elif structure[i] == ')':
            prob += structure_alignment[2,i]
        

    return prob / len(structure)

#-----------------------------------------------
def create_SANDSTORM_cas(seq_len=60,ppm_len=60,latent_dim=128,internal_activation='relu',output_activation='linear',output_nodes=1,output_units=1,kernel_1_size=[8,18],kernel_2_size=[8,9],kernel_3_size=[8,3]):
    # seq_len is the length of the sequences
    # Latent dim controls the number of filters
    # internal activation is a tensorflow activation function, applied to the hiden layers
    # output activation is the activation function of the model output
    # output nodes is the number of different output channels the model will have (e.g. 1 for UTR prediciton, 2 for Toehold Prediction (on, off)
    # output units is the number of units in the output layer, 1 for regression problems and N for classification with N different classes 
    


    #Define the Inputs
    input_seqs = tfk.Input(shape=(8,seq_len,1))
    input_probs = tfk.Input(shape=(ppm_len,ppm_len,1))


    # Predictive Model Definition
    #Sequence Branch
    y = tfkl.Conv2D(latent_dim/4, kernel_1_size, strides=(8, 1), padding="same",activation=internal_activation)(input_seqs)
    # y = tfkl.SpatialDropout2D(0.2)(y)
    y = tfkl.BatchNormalization()(y)
    y = tfkl.Conv2D(latent_dim/8, kernel_2_size, strides=(8, 1), padding="same",activation=internal_activation)(y)
    y = tfkl.Conv2D(latent_dim/16,kernel_3_size,strides=(8,1),padding='same',activation=internal_activation)(y)
    y = tfkl.Flatten()(y)

    #PPM Branch    
    x = tfkl.Conv2D(latent_dim/4, (9, 9), strides=(ppm_len, 1), padding="same",activation=internal_activation)(input_probs)
    # x = tfkl.SpatialDropout2D(0.2)(x)
    x = tfkl.SpatialDropout2D(0.2)(x)
    x = tfkl.Conv2D(latent_dim/8, (5, 5), strides=(ppm_len, 1), padding="same",activation=internal_activation)(x)
    x = tfkl.Conv2D(latent_dim/16,(3,3),strides=(ppm_len,1),padding='same',activation=internal_activation)(x)
    x = tfkl.GlobalMaxPooling2D()(x)
    x = tfkl.Flatten()(x) 



    #Combine the two
    z = tfkl.Concatenate()([x,y])
    z = tfkl.Dense(16,activation=internal_activation)(z)
    z = tfkl.Dense(8,activation=internal_activation)(z)
    z = tfkl.Dense(4,activation=internal_activation)(z)
	
    
    output_lst = []
    for i in range(output_nodes):
        tmp = tfkl.Dense(output_units,activation=output_activation,name='prediction_output_%s'%i)(z)
        output_lst.append(tmp)


    output_model  = tfk.Model(inputs=[input_seqs,input_probs],outputs=output_lst,name='joint_model')
    return output_model