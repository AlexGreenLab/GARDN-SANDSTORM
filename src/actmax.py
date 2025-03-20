"""

Module containing methods for optimizing single sequences and GANs against a predictor


@author: aidan 
"""

import tensorflow as tf
import nupack as n
import tf_agents
import tensorflow.keras as keras
from GA_util import create_ppms_fast
import numpy as np
from GA_util import GumbelSoftmax
from util import unencode
import matplotlib.pyplot as plt
import util
import GA_util
import time
import pandas as pd



#------------------------------------------------
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

import logging
logging.getLogger().setLevel(logging.ERROR)
EPSILON = 1e-20



#------------------------------------------------
class G(Layer):
    def __init__(self, axis: Optional[int] = 1, **kwargs) -> None:
        super(G, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor):
        #Inputs will be the output of a generator, shape = BATCH_SIZE, 4, SEQ_LEN,1
        # or they will be a sequence tensor that is shape BATCH_SIZE,4,SEQ_LEN
        if inputs.shape[-1] == 1:
            inputs = tf.squeeze(inputs,axis=3)
            
        gumbel_dist = tf_agents.distributions.gumbel_softmax.GumbelSoftmax(temperature=1000,logits=tf.transpose(inputs,(0,2,1)))
        sample = tf.transpose(gumbel_dist.sample(),(0,2,1))
        
        return tf.nn.softmax(inputs + sample,axis=1)
    
#------------------------------------------------  
class G_valeri(keras.layers.Layer):
    def __init__(self, axis= 1):
        super(G_valeri, self).__init__()

    def call(self, inputs):
        #Inputs will be the output of a generator, shape = BATCH_SIZE, seq_len,4
        gumbel_dist = tf_agents.distributions.gumbel_softmax.GumbelSoftmax(temperature=1000,logits=inputs)
        sample = gumbel_dist.sample()
        
        return tf.nn.softmax(inputs + sample,axis=2)
    
    
    
    
    
    
#------------------------------------------------
class ActMaxModel(keras.Model):
    """Combines the generator and predictor into an end-to-end model for training."""

    def __init__(self,generator_model,predictor_model,UTR=True):
        
        super(ActMaxModel, self).__init__()

        self.generator_model = generator_model
        self.predictor_model = predictor_model

    def call(self, inputs):
        seqs = self.generator_model(inputs) # Create a generated sequence
        # softmax_seqs = G()(seqs)
       
        ppms = GA_util.prototype_ppms_fast(seqs[:,:,:,0]) # Calculate the PPM
        defect = self.predictor_model([seqs,ppms])# Return the predicted ON
        return defect
    
#------------------------------------------------   
class ActMaxModel_NG(keras.Model):
    """Create an ActMaxModel that optimizes sequence-inputs, not a generator"""

    def __init__(self,predictor_model):
        
        super(ActMaxModel_NG, self).__init__()

        self.predictor_model = predictor_model

    def call(self, inputs):
        #Inputs are one-hot-encode sequence vectors of shape BATCH_SIZE,4,SEQ_LEN
        
        # softmax_seqs = G()(inputs) #pass it through the distribution
        softmax_seqs = inputs
        # softmax_seqs = softmax_seqs[:,:,:] 
       
        ppms = GA_util.prototype_ppms_fast(softmax_seqs) # Calculate the PPM
        defect = self.predictor_model([softmax_seqs,ppms])# Return the predicted ON
        return defect
    
    
#------------------------------------------------
class ActMaxModel_valeri(keras.Model):

    def __init__(self,predictor_model):
        
        super(ActMaxModel_valeri, self).__init__()
        self.predictor_model = predictor_model

    def call(self, inputs):
        softmax_seqs = G_valeri()(inputs)
       
        defect = self.predictor_model(inputs)[0]# Return the predictd ON/OFF Ratio 
        return defect
    
#------------------------------------------------      
class ActMaxModelAptaswitch(keras.Model):
    """Combines the generator and predictor into an end-to-end model for training."""

    def __init__(self,generator_model,predictor_model,UTR=True):
        
        super(ActMaxModelAptaswitch, self).__init__()

        self.generator_model = generator_model
        self.predictor_model = predictor_model
        
        
        #This model needs to glue context to the toehold to make it an aptaswitch
        self.core = 'GUCGAGUAGAGUGUGGGCUCAGAUUCGUCUGAGACGGUCGGGUCC'
        self.core = self.core.replace("U",'T')
        self.promoter = 'GCGCTAATACGACTCACTATAGGG'
        self.AA = 'AA'

        self.core = util.one_hot_encode_str(self.core)
        self.promoter = util.one_hot_encode_str(self.promoter)
        self.AA = util.one_hot_encode_str(self.AA)

        self.core = tf.cast(tf.expand_dims(self.core,axis=-1),tf.float32)
        self.promoter = tf.cast(tf.expand_dims(self.promoter,axis=-1),tf.float32)
        self.AA = tf.cast(tf.expand_dims(self.AA,axis=-1),tf.float32)

    def call(self, inputs):
        seqs = self.generator_model(inputs) # Create a generated sequence
        # softmax_seqs = G()(seqs)
        #Glue the aptaswitch context
        seqs = seqs[:,:,1:] #Seqs will be nseqs,4,60, so slice out this part
        # print(seqs.shape)
        
        b_region = seqs[:,:,12:19] #This is only valid if the switch is 59 nucleotides
        # print(b_region.shape)
        
        
        seqs = tf.concat((self.promoter,seqs,self.core,b_region,self.AA),axis=2)
        # print(seqs.shape)
        
        ppms = GA_util.prototype_ppms_fast(seqs[:,:,:,0]) # Calculate the PPM
        defect = self.predictor_model([seqs,ppms])# Return the predicted ON
        return defect
    
    def get_aptaswitch(self,inputs):
        """returns the aptaswitch conformation from random latent vector inputs"""
        
        seqs = self.generator_model(inputs) # Create a generated sequence
        seqs = seqs[:,:,1:]
        # print(seqs)
        b_region = seqs[:,:,12:19]

        # softmax_seqs = G()(seqs)
        #Glue the aptaswitch context
        
        promoter_stack = np.stack([self.promoter[0,:,:,0] for i in range(inputs.shape[0])])
        core_stack = np.stack([self.core[0,:,:,0] for i in range(inputs.shape[0])])
        AA_stack = np.stack([self.AA[0,:,:,0] for i in range(inputs.shape[0])])
        

        seqs = np.concatenate((promoter_stack,seqs[:,:,:,0],core_stack,b_region[:,:,:,0],AA_stack),axis=2)
        # print(seqs.shape)
        return seqs
    
    def get_triggers(self,inputs):
        """returns the aptaswitch conformation from random latent vector inputs"""
        
        seqs = self.generator_model(inputs) # Create a generated sequence
        seqs = seqs[:,:,:]
        triggers = seqs[:,:,:30]
        
        return GA_util.revcompmult(triggers.numpy())
    
    
    
    
#------------------------------------------------
def calc_loss(vector, model,output=None):

    # convert to batch format
    vector_batch = tf.expand_dims(vector, axis=0)
    # run the model
    if output == 'on':
        layer_activations = model(vector_batch)[0]
    elif output == 'off':

        layer_activations = model(vector_batch)[1]
        
    elif output =='UTR_high':
        layer_activations = model(vector_batch)[0]
    elif output =='UTR_low':
        layer_activations = model(vector_batch)[0]
    elif output =='MFE':
        layer_activations = model(vector_batch)[0]
    else:
        raise Exception('Accepted output values are "on","off","UTR_high",or "UTR_low"')



    return layer_activations

#------------------------------------------------
class ActMaxOptimizer(tf.Module):
    
    def __init__(self,model,output):
        self.model = model
        self.output = output
        # print(self.output)

    def __call__(self,vector,steps,step_size):
    
        # print('Improving...')

        # loss = tf.constant(0.0)
       
        for n in tf.range(steps):
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                
                tape.watch(vector)
                
                loss = calc_loss(vector,self.model,self.output)
                
            
            gradients = tape.gradient(loss,vector)
            

            if self.output == 'on' or self.output == 'UTR_high':

                vector = vector + (gradients*step_size)
            elif self.output == 'off' or self.output == 'UTR_low' or self.output=='MFE':
                
                vector = vector - (gradients*step_size)
            
            
        return loss,vector
    
    

    
#------------------------------------------------ 
def run_act_max_single_seq(vector,act_max_mod,n_steps=100,step_size=0.01,output='on'):
    """function that facilitates the actual actmax process
    vector: can be either a random one-hot-encoded sequence or a particular starting sequence
    ActMax_model: an instance of ActMaxModel or ActMaxModel_NG class
    ActMax_optimizer: and instance of ActMax_optimizer class
    steps: how many updates to pass the sequence through before terminating
    step_size: alpha factor multiplying by the gradient after each update
    output: None for now, could be adjusted to make the index of outupt to optimize for multi-output models 
    """
    opt = ActMaxOptimizer(act_max_mod,output=output)
    
    on_save = []
    seq_save = []
    vector = tf.Variable(vector,trainable=True)

    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = n_steps
    step = 0
    while steps_remaining:
        print(steps_remaining)
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
  
        loss,vector = opt(vector,run_steps,tf.constant(step_size))
        
        
    on_save.append(loss)
    seq_save.append(unencode(vector))



    return vector,seq_save,on_save #This is not returning an updated vector

#------------------------------------------------ 
def run_act_max_mult_seqs(act_max_mod,starting_seq,n_seqs=300,output='on',n_steps=300,step_size=0.01,verbose=True):

    dat = np.zeros(shape=(nseqs,6))
    df = pd.DataFrame(dat,columns=['Pre_Optimized_Sequence','Post_Optimized_Sequence','Pre_Optimized_Value','Post_Optimized_Value','Target_Value','Delta'])
        

    for i in range(nseqs):
        iteration_start = time.time()
        output = 'on'


        ipt_seq = tf.cast(starting_seq,'float32')
        
        
        if verbose:
            print('Start Sequence %s'%i)
            print(unencode(ipt_seq[0,:,:]))
        
        max_seq,seq_save,on_save = run_act_max_single_seq(vector=ipt_seq,act_max_mod=act_max_mod, n_steps=n_steps, step_size=step_size,output=output)
                  
        on_save = [i.numpy() for i in on_save]
        

        df.iloc[i,0] = seq_save[0] #append the pre optimization sequence
        df.iloc[i,1] = seq_save[-1] #append the final optimized sequence
        df.iloc[i,2] = on_save[0]
        df.iloc[i,3] = on_save[-1]
        df.iloc[i,4] = output
        df.iloc[i,5] = on_save[-1] - on_save[0] # Final - Initial
        
        if verbose:
            print(unencode(max_seq))
            print('iteration %s took %.2f seconds'%(i,time.time()-iteration_start))
        
        
    return df



#------------------------------------------------
# from GA_util import unencode
# from actmax import ActMaxOptimizer
def run_gan_opt_single_seq(vector,act_max_mod,steps=100,step_size=0.01,output='on',generator=True):
    """function that facilitates the actual GARDN optimization process
    vector: can be either a random one-hot-encoded sequence for single-sequence act max or a random noise if generator optimization
    ActMax_model: an instance of ActMaxModel or ActMaxModel_NG class
    ActMax_optimizer: and instance of ActMax_optimizer class
    steps: how many updates to pass the sequence through before terminating
    step_size: alpha factor multiplying by the gradient after each update
    output: None for now, could be adjusted to make the index of outupt to optimize for multi-output models
    """

    opt = ActMaxOptimizer(act_max_mod,output=output)
    vector = tf.Variable(vector,trainable=True)

    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        # print(steps_remaining)
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        
                    
        loss,vector = opt(vector,run_steps,tf.constant(step_size))



    return vector

#------------------------------------------------ 
def run_gan_opt_mult_seqs(act_max_mod,nseqs=300,output='on',n_steps=300,step_size=0.1,verbose=True,latent_dim=128,starting_seq=None):
    dat = np.zeros(shape=(nseqs,4))
    df = pd.DataFrame(dat,columns=['Pre_Optimized_Value','Post_Optimized_Value','Target_Value','Delta'])
    
    
    vector_output = np.zeros((nseqs,latent_dim,2)) #nseqs, latent_dim, 2 channels for starting and stopping sequence
    
    for i in range(nseqs):
        iteration_start = time.time()
        # output = 'on'

        
        if starting_seq == None:
            random_latent_vectors = tf.cast(np.random.normal(size=(latent_dim,)),'float32')
        # print('starting seq is',starting_seq)
        else:
            random_latent_vectors = starting_seq
        

        
        
        if verbose:
            print('Start Sequence %s'%i)
            # print(unencode(ipt_seq))
            
        vector_output[i,:,0] = random_latent_vectors#append the pre optimization latent variable    
        
        

        start_val = act_max_mod(tf.expand_dims(random_latent_vectors,axis=0))
        if type(start_val) == list:
            start_val = start_val[0].numpy()[0][0]
        else:
            start_val = start_val.numpy()[0][0]

        
        vector = run_gan_opt_single_seq(vector=random_latent_vectors,act_max_mod=act_max_mod, steps=n_steps, step_size=step_size,output=output)
        # on_save = [i.numpy() for i in on_save]
        
        stop_val = act_max_mod(tf.expand_dims(vector,axis=0))
        if type(stop_val) == list:
            stop_val = stop_val[0].numpy()[0][0]
        else:
            stop_val = stop_val.numpy()[0][0]
        
        if verbose:
            print('Start val: ',start_val)
            print('Stop val: ',stop_val)
        

        # print(on_save)
        df.iloc[i,0] = float(start_val)
        df.iloc[i,1] = float(stop_val)
        df.iloc[i,2] = output
        # print(on_save[-1] - on_save[0])
        df.iloc[i,3] = start_val - stop_val # Final - Initial
        

        vector_output[i,:,1] = vector #append the final optimized latent variable

        if verbose:
            print('iteration %s took %.2f seconds'%(i,time.time()-iteration_start))
            print('\n')
        
        
    return vector_output,df




#Helper functions for activaiton maximization using the transposed sequence format found in the Valeri et al. Paper

def create_rand_valeri_inputs(n_seqs):
    out = np.zeros(shape=(n_seqs,59,4))
    for i in range(n_seqs):
        for j in range(59):
            nt = GA_util.get_random_nt()
            out[i,j,GA_util.nucleotides[nt]] = 1
    return out

def unencode_valeri(sequence):
    out = ''
    # print(sequence.shape)
    for i in range(sequence.shape[0]):
        
        val = np.argmax(sequence[i,:])
        out += GA_util.letters[val]
    return out  

def calc_loss_val_model(vector, model):
    # print('calc loss:',vector.shape)
    #Maximizing the 'ON' output prediction
    layer_activations = model(vector)[0]

    return layer_activations

def constrained_rand_valeri_inputs(n_seqs):
    #here we are providing the start codon and RBS to the random input
    
    triggers = create_rand_valeri_inputs(n_seqs)
    print(triggers.shape)
    
    rbs = np.array([[1.0,0.0,0.0,0.0],
           [1.0,0.0,0.0,0.0],
           [0.0,0.0,1.0,0.0],
           [1.0,0.0,0.0,0.0],
           [0.0,1.0,0.0,0.0],
           [1.0,0.0,0.0,0.0],
           [0.0,1.0,0.0,0.0],
           [0.0,1.0,0.0,0.0],
           [1.0,0.0,0.0,0.0],
           [0.0,1.0,0.0,0.0],
           [1.0,0.0,0.0,0.0]])

    
    
    atg = np.array([[1.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,1.0],
                    [0.0,1.0,0.0,0.0]])
    
    triggers[:,30:41,:] = rbs
    triggers[:,47:50,:] = atg
    
    return triggers