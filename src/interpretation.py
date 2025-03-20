"""
Created on Fri June 9 20:28:48 2023

@author: aidan

Module for various DL interpretation techniques and visualizations
"""
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import tensorflow as tf
import GA_util
#-----------------------------------------------
def plot_kernel_logo(kernel,title=None,figsize=None):
    #kernel should be the weights of your tf model conv layer
    
    if figsize is None:
        figsize = [10,2]
        
        
    fig, ax = plt.subplots(1,1,figsize=figsize)

    kernel = np.transpose(kernel,(1,0))
    

    kernel_df = pd.DataFrame(data = kernel,columns = ['A', 'G', 'C', 'U'])
    

    nn_logo = logomaker.Logo(kernel_df,
                             ax=ax,
                            shade_below=.5,
                            fade_below=.5,
                             font_name='Arial Rounded MT Bold',color_scheme='colorblind_safe')
    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left'], visible=True)

    # style using Axes methods
    
    nn_logo.ax.set_xticks([])
    
    
    
    nn_logo.ax.set_ylabel('Kernel Results', labelpad=-1)
    
    if title != None:
        crp_logo.ax.set_title(title,font='Helvetica',fontsize=12)
 
#-----------------------------------------------    
def compute_gradients(model,sequences,ppms=None,return_val=None,baseline=None):
    #If no structure matrix is passed, calculate one based on the baseline sequence
    #return val can be 'seq', 'ppm', or None; None defaults to sequence returns
    #baseline optional arugment to pass a custom baseline sequence instead of zero vector
    
    if return_val is None:
        return_val = 'seq'
    
    if ppms is None:
        #Need to select the -1 sequence specifically because it will be the ground truth while priors are interpolations
        ppms = GA_util.create_ppms_fast(np.expand_dims(sequences[-1,:,:],axis=0))

        ppms = interpolate_ppm(ppms,m_steps = sequences.shape[0]-1,baseline=baseline)
    
    
    if return_val == 'seq':
        with tf.GradientTape() as tape:
            tape.watch(sequences)
            output = model([sequences,ppms])
        return tape.gradient(output, sequences)
    elif return_val == 'ppm':
        with tf.GradientTape() as tape:
            tape.watch(ppms)
            output = model([sequences,ppms])
        return tape.gradient(output,ppms)

#-----------------------------------------------  
def interpolate_sequence(sequence,baseline=None,m_steps=50):
    #modification of the original fucntion to accept different baseline sequences
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) 
    
    sequence = sequence.astype('float')
    if baseline is None:
        baseline=np.zeros(shape=sequence.shape,dtype='float')
    
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(sequence, axis=0)
    
    baseline_x,input_x = tf.cast(baseline_x,'float'),tf.cast(input_x,'float')
    delta = input_x - baseline_x
    
    
    sequences = baseline_x +  alphas_x * delta
    return sequences


#-----------------------------------------------  
def interpolate_ppm(ppm,baseline=None,m_steps=50):
    #creates m_steps linear interpolations of the ppm corresponding to a sequence to a 0-vector baseline of the same shape
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) 
    
    # ppm = GA_util.create_ppms_fast(np.expand_dims(sequence,axis=0))
    # ppm = ppm.astype('float')
    
    # sequence = sequence.astype('float')
    if baseline is None:
        baseline=np.zeros(shape=ppm.shape,dtype='float')
    
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    # baseline_x = 
    # input_x = tf.expand_dims(ppm, axis=0)
    
    baseline_x,input_x = tf.cast(baseline,'float'),tf.cast(ppm,'float')
    delta = input_x - baseline_x
    
    
    output_ppms = baseline_x +  alphas_x * delta
    return output_ppms

#-----------------------------------------------   
def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

#-----------------------------------------------   
def predict_swap(sequence,model):
    #sequence is a one hot encoded sequence, shape 4,SEQ_LEN
    #returns an list of length 4,SEQ_LEN, where each row is the trace of activity for each other nt
    #the average of this array is the total delta of a nt swap
    
    output = np.zeros((4,sequence.shape[1]))
    print(output.shape)
    
    og_pred = model.predict([tf.expand_dims(sequence,axis=0),tf.expand_dims(GA_util.pairwise_prob_fastest(sequence),axis=0)],verbose=0)[0]
    print(og_pred)
    
    #Start by getting all the rows that need to be swapped
    for i in range(sequence.shape[1]):
    
        possible_rows = [0,1,2,3]
        row_to_swap = np.argmax(sequence[:,i])
        remaining = [i for i in possible_rows if i != row_to_swap]
        
        
        
        for j in remaining:
            #zero everythign in this spot out
            sequence[:,i] = 0
            
            #Make the value at one specific row 1, representing a swap
            sequence[j,i] = 1
            #create the ppm of the corresponding sequence
            tmp_ppm = tf.expand_dims(GA_util.pairwise_prob_fastest(sequence),0)
            #predict the activity
            tmp_pred = model.predict([tf.expand_dims(sequence,axis=0),tmp_ppm],verbose=0)[0]
            #save the activity into the output
            output[j,i] = og_pred - tmp_pred
                        
        ###revert back to og before moving to next position on sequence length
        sequence[row_to_swap,i] = 1
        output[row_to_swap,i] = og_pred - model.predict([tf.expand_dims(sequence,axis=0),tf.expand_dims(GA_util.pairwise_prob_fastest(sequence),axis=0)],verbose=0)[0]
        
    return output

#----------------------------------------------- 
def compute_gradients_classifier(model,sequences,ppms=None,return_val=None,target_class=0):
    #If no structure matrix is passed, calculate one based on the baseline sequence
    #return val can be 'seq', 'ppm', or None; None defaults to sequence returns
    #target_class is an integer indexign which model output you are interested in
    
    if return_val is None:
        return_val = 'seq'
    
    if ppms is None:
        #Need to select the -1 sequence specifically because it will be the ground truth while priors are interpolations
        ppms = GA_util.create_ppms_fast(np.expand_dims(sequences[-1,:,:],axis=0))

        ppms = interpolate_ppm(ppms,m_steps = sequences.shape[0]-1)
    
    
    if return_val == 'seq':
        with tf.GradientTape() as tape:
            
            tape.watch(sequences)
            output = model([sequences,ppms])[:,target_class]
            
        return tape.gradient(output, sequences)
    
    
    elif return_val == 'ppm':
        with tf.GradientTape() as tape:
            tape.watch(ppms)
            output = model([sequences,ppms])[:,target_class]
        return tape.gradient(output,ppms)


#-----------------------------------------------
# Function to compute gradients
def compute_gradients_rand_inputs(model, dummy_input):
    with tf.GradientTape() as tape:
        tape.watch(dummy_input)
        output = model(dummy_input)
    gradients = tape.gradient(output, dummy_input)
    return gradients[0],gradients[1]
    # return gradients.numpy().flatten()

#-----------------------------------------------
# Function to compute Hessian eigenvalues (largest eigenvalue estimation)
def compute_hessian_eigenvalue_rand_inputs(model, dummy_input):
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(dummy_input)
            output = model(dummy_input)
        gradients = tape1.gradient(output, dummy_input)
    hessian = tape2.jacobian(gradients, dummy_input).numpy().reshape((input_shape[1], input_shape[1]))
    eigenvalues = np.linalg.eigvalsh(hessian)  # Get Hessian eigenvalues
    return np.max(np.abs(eigenvalues))  # Return the largest absolute eigenvalue