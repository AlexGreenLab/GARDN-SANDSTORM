# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:01:33 2022

module for basic data loading and one-hot-encoding
probably needs to be re-organized

@author: aidan
"""

#Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import warnings
import seaborn as sns

import nupack as n
# import matplotlib.pyplot as plt
# import numpy as np

     
import tensorflow
tfk = tensorflow.keras
tfkl = tfk.layers
import keras
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


#-----------------------------------------------
#Save the nucleotide order for encoding/indexing
nucleotides = {'A':0,'G':1,'C':2,'T':3,'a':0,'g':1,'c':2,'t':3,'U':3}
letters = ['A','G','C','T']

#matplotlib color saves
RED = '#C62626'
BLUE = '#2E9AD6'


#-----------------------------------------------
#Replace substring fucntion
def replace(str1, str2):
    out = str1.replace(str2, '')
    return out




#-----------------------------------------------
#Load dataset from Chen et al. 

#Need to add option to remove poly A and U track 

def load_data(path,filter_poly_track=False):
    #Path should be copied and pasted as the directory where the xlsx file is stored
    out = pd.read_excel(path)
    strength = out['Average Strength']
    seqs = out['Sequence'].str.upper()
    if filter_poly_track:
        poly_a = out['A-tract'].str.upper()
        poly_u = out['U-tract'].str.upper()
        for i in range(seqs.shape[0]):
            poly_a[i] = poly_a[i].replace('U','T')
            poly_u[i] = poly_u[i].replace('U','T')

            seqs[i] = replace(seqs[i],poly_u[i])
            

            

    seqs = seqs.apply(lambda x: pd.Series(list(x)))
    return seqs.to_numpy(), strength.to_numpy()



#-----------------------------------------------
#Find longest
def longest_length(seqs):
    #seqs is shape nseq, seq_len
    out = seqs.shape[1]
    return out



#-----------------------------------------------    
#One Hot Encoding

# Since sequences are different lengths, this will also zero pad the shorter ones

def one_hot_encode(seqs):
    full_set = seqs.sum(axis=1).astype(str)
    seqs = full_set.apply(lambda x: pd.Series(list(x))).to_numpy()

    #seqs is shape (nseqs,seq_len)
    nseqs = seqs.shape[0]
    seq_len = seqs.shape[1]
    out = np.zeros(shape = (nseqs,4,seq_len))
    for i in range(nseqs):
        for j in range(seq_len):
            #Nan catch
            if type(seqs[i,j]) is float:
                pass
            #Catch for a typo with spaces in line 26 of the natural terminator dataset
            elif seqs[i,j] == ' ':
                seqs[i,:] = np.roll(seqs[i,:],shift=-1,axis=0)
                seqs[i,-1] = np.nan
            else:
                if seqs[i,j] == 'Z':
                    pass
                else:
                    idx = nucleotides[seqs[i,j]]
                    out[i,idx,j] = 1
    return out


#-----------------------------------------------
def one_hot_encode_str(sequence):
    #Sequence shoudl be an ATGC or AUGC string, output will be shape 1,4,seq_len
    nucleotides = {'A':0,'G':1,'C':2,'T':3,'a':0,'g':1,'c':2,'t':3,'U':3,'u':3 }
    
    output = np.zeros(shape=(1,4,len(sequence)))
    
    idx_lst = [nucleotides[i] for i in sequence]

    rows = np.arange(4)  # Array of row indices [0, 1, 2, 3]
    mask = rows[:, np.newaxis] == np.array(idx_lst)[np.newaxis, :]

    # Assign value 1 to the selected rows using the mask
    output[:, mask] = 1
    
    return output

#-----------------------------------------------
def one_hot_encode_str_lst(seq_lst):
    #Seq lst is a list of sequences, where each sequence is a string in alphabet AGCT or AGCU
    #seq lst can contain sequences of different lengths and the output will be the max length in the list with
    #all shorter sequences 0-padded on the right
    
    seq_len = len(max(seq_lst))
    
    output = np.zeros(shape=(len(seq_lst),4,seq_len))
    
    for i,seq in enumerate(seq_lst):
        
        output[i,:,:] = one_hot_encode_str(seq)
        
    return output

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
def activate(seq):
    #seq is shape 4, seq_len
    out = np.zeros_like(seq)
    idx = np.argmax(seq,axis=0)
    

    out[idx,np.arange(0,len(idx))] = 1
    return out




#-----------------------------------------------
def unencode(sequence):
    out = ''
    for i in range(sequence.shape[1]):
        val = np.argmax(sequence[:,i])
        out += letters[val]
    return out  


#-----------------------------------------------
#Load and encode data in 1 step
def load_and_encode(path,filter_poly_track=False):
    (seqs,strength) = load_data(path,filter_poly_track)
    seqs = one_hot_encode(seqs)
    return seqs,strength



#-----------------------------------------------
#Stack the sequence arrays
def stack_seqs(seqs_1,seqs_2):
    #Stacking 2 different size arrays of shape (nseqs,4,seq_len)
    max_dim = max(seqs_1.shape[2],seqs_2.shape[2])
    out = np.zeros(shape = (seqs_1.shape[0] + seqs_2.shape[0],4,max_dim))
    
    out[:seqs_1.shape[0],:,:seqs_1.shape[2]] = seqs_1[:,:,:]
    out[seqs_1.shape[0]:,:,:seqs_2.shape[2]] = seqs_2[:,:,:]
    
    return out


#-----------------------------------------------
#Create probability weight matrix
#pwms are positions in rows and frequencies of each letter in columns
#seqs are nseqs,4,seq_len
def create_pwm(seqs):

    out = np.zeros(shape=(seqs.shape[2],4))
    locs = np.argmax(seqs,axis=1)

    
    
    tot = seqs.shape[0] #denominator value
    for i in range(seqs.shape[2]):   
        a_num = np.sum(locs[:,i] == 0) / tot
      
        g_num = np.sum(locs[:,i] == 1) / tot
    
        
        c_num = np.sum(locs[:,i] == 2) / tot
     
        
        t_num = np.sum(locs[:,i] == 3) / tot
   
        out[i,0] = a_num
        out[i,1] = g_num
        out[i,2] = c_num
        out[i,3] = t_num
    return out
    

#-----------------------------------------------
#Plot a pwm
def plot_logo(seqs,title=None,figsize=None):
    if figsize is None:
        figsize = [10,2]
        
    pwm = create_pwm(seqs)
    # create Logo object
    pwm = pd.DataFrame(pwm, columns = ['A','G','C','U'])
    
    fig, ax = plt.subplots(1,1,figsize=figsize)

    crp_logo = logomaker.Logo(pwm,
                              ax=ax,
                              shade_below=.5,
                              fade_below=.5,
                              font_name='Arial Rounded MT Bold',
                              color_scheme='colorblind_safe')

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    # crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    crp_logo.ax.set_ylabel("Frequency", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    # crp_logo.ax.xaxis.set_tick_params(pad=-1)
    # crp_logo.ax.xaxis.set_xticks([])
    crp_logo.ax.spines['left'].set_linewidth(2.0)
    crp_logo.ax.spines['bottom'].set_linewidth(2.0)
    
    if title != None:
        crp_logo.ax.set_title(title,font='Helvetica',fontsize=12)
        
#-----------------------------------------------       
def single_seq_logo(seq):
    out = np.reshape(seq,(1,seq.shape[0],seq.shape[1]))
    plot_logo(out)

#-----------------------------------------------   
def plot_kernel_logo(kernel):
    #kernel should be the weights of your tf model conv layer

    kernel = np.reshape(kernel,(kernel.shape[1],4))

    kernel_df = pd.DataFrame(data = kernel,columns = ['A', 'G', 'C', 'T'])
    

    nn_logo = logomaker.Logo(kernel_df,font_name='Arial Rounded MT Bold',
                              color_scheme='colorblind_safe')
    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left'], visible=True, bounds=[0, .75])

    # style using Axes methods
    
    nn_logo.ax.set_xticks([])
    
    
    
    nn_logo.ax.set_ylabel('Kernel Results', labelpad=-1)
    
    
#-----------------------------------------------   
def print_copyable(seqs):
    print([list(i) for i in seqs])
    
 #-----------------------------------------------    
def load_collins_data(path ='data/Toehold_Dataset_Final_2019-10-23.csv', switch_only = True,threshold=None,return_values=False,val='ON'):
    # #%%Load and encode Collins data
    
    ####Currently does not return any ON/OFF values, just the sequences themselves
    data = pd.read_csv(path)

    data = data[data[val].notna()]
    
    if threshold is not None:
        data = data[data[val] > threshold]
        values = data[val]
    
    if switch_only:
        full_set = data[['switch','loop2','stem1','atg','stem2']]
        full_set = full_set.sum(axis=1).astype(str)
        
        #Adding the constant previous C nucleotide to make switches 60 nt long
        full_set = 'C' + full_set
        seqs = full_set.apply(lambda x: pd.Series(list(x))).to_numpy()

        seqs = one_hot_encode(seqs)
        
    else:
        full_set = data[['pre_seq','promoter','trigger','loop1','switch','loop2','stem1','atg','stem2','linker','post_linker']]
        # full_set = full_set.sum(axis=1).astype(str)
        # seqs = full_set.apply(lambda x: pd.Series(list(x))).to_numpy()

        seqs = one_hot_encode(full_set)       
    if return_values:
        return seqs,values
    else:
        return seqs
    
 #-----------------------------------------------    

def load_valeri_data(path ='data/Toehold_Dataset_Final_2019-10-23.csv', threshold=None):
    # #%%Load and encode Collins data
    
    ####Currently does not return any ON/OFF values, just the sequences themselves
    data = pd.read_csv(path)

    data = data[data['ON'].notna()]
    data = data[data['OFF'].notna()]
    
    data = data[data['ON'] > threshold]
    data = data[data['OFF'] > threshold]
    on = data['ON']
    off = data['OFF']

    full_set = full_set = data[['switch','loop2','stem1','atg','stem2']]
    # full_set = full_set.sum(axis=1).astype(str)
    # seqs = full_set.apply(lambda x: pd.Series(list(x))).to_numpy()

    seqs = one_hot_encode(full_set)       

    return seqs,on,off





    
#-----------------------------------------------   
def neat_plot(sp=None):
    # plt.rcParams.update(['axes.axisbelow'] = True
    # plt.subplot().set_facecolor('#EBEBEB')
    # plt.grid(alpha=0.5)
    ggplot_styles = {
    'font.family': 'Helvetica',
    'font.size':12,'axes.axisbelow':True}

    plt.rcParams.update(ggplot_styles)

    #sp is a subplot index
    if sp is None:
        plt.subplot().spines['left'].set_linewidth(2.0)
        plt.subplot().spines['bottom'].set_linewidth(2.0)
        plt.subplot().spines['right'].set_visible(False)
        plt.subplot().spines['top'].set_visible(False)
        
    else:
        plt.subplot(sp).spines['left'].set_linewidth(2.0)
        plt.subplot(sp).spines['bottom'].set_linewidth(2.0)
        plt.subplot(sp).spines['right'].set_visible(False)
        plt.subplot(sp).spines['top'].set_visible(False)
        



#-----------------------------------------------
def plot_regression_output(model,input_data,output_data,color='cyan',title='Regression Results',alpha=0.6):
    #Model is a trained tf model
    #input data is a list of inputs to said model (Joint model with 2 inputs in mind)
    #output data is the true regression values, y_test
    
    preds = model.predict(input_data) #calling just model(data) crashes the kernel weirdly
    preds = preds.reshape(preds.shape[0])
    # r2 = r2_score(output_data,preds)
    spearman = spearmanr(output_data,preds)[0]
    
    plt.figure()
    plt.scatter(output_data,preds,color=color,alpha=alpha)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.title('%s Spear.=%.2f'%(title,spearman))
    
#-----------------------------------------------  
def make_regression_model(inputs,activation_function='linear',name=''):
    seq_input = tfk.Input(shape=[inputs.shape[1],inputs.shape[2]])
    output = tfkl.Conv1D(1,kernel_size=(inputs.shape[1]),activation=activation_function)(seq_input)
    # output = tfkl.Dense(1,activation=activation_function)(seq_input)
    model = tfk.Model(inputs=seq_input, outputs=output, name="regression_%s"%(activation_function))
    
    return model

#---------------------------------------

def train_regression_model(input_model,input_data,output_data,LOSS='mse',EPOCHS=20,BATCH_SIZE=1):
    #returns a trained_model
    #input data should be a list of format [seq_array,output_variable]
    output_data = np.array(output_data)
    opt = keras.optimizers.Adam(lr=0.001)
    input_model.compile(optimizer=opt,loss=LOSS)
    input_model.fit(input_data,output_data,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=False)
    
    return input_model

#---------------------------------------
def get_data_slice(dataframe,category,search_column,sep=None,sep_idx=None,slice_idx=None):
    #Funciton for slicing a dataset and returning all elements in a specific category
    #category is the substring that should identify members of the dataset to keep
    #search_column is the pandas df column that holds the categories 
    #optional argument sep and sep_idx will split the entries in search column based on this character to look for the substring at sep index
    #sometimes you want to compare the unifying substring known as category to a slice of the entires in the search column,
    # This can be done using the slice_idx variable which will change the behavior to check if 
    
    idx_save = []
    
    if sep is None:
        for i,name in enumerate(dataframe[search_column]):
            if slice_idx is None:
                if category in name:
                    idx_save.append(i)
            else:
                if category == name[:slice_idx]:
                    idx_save.append(i)
            
    else:
        for i,name in enumerate(dataframe[search_column]):
            split_list = name.split(sep) #Split by the provided character
            
            if slice_idx is None:
                if split_list[sep_idx] == category:
                    idx_save.append(i)
            else:
                if split_list[sep_idx][:slice_idx] == category:
                    idx_save.append(i)
                    
    output = dataframe.iloc[idx_save,:]
    return output


#---------------------------------------
def model_plot(ground_truth,predictions):
    g = sns.jointplot(x=ground_truth,y=predictions,kind='scatter')
    g.plot_joint(sns.kdeplot, color="darkblue", zorder=1, levels=6)
    g.plot_joint(sns.regplot,scatter=False,color='black')
    plt.xlabel('Measured Function')
    plt.ylabel('Predicted Function')
    
#---------------------------------------    
def dotparen_to_pairs(ipt):
    stack_one = []  # Stack to store indices of opening parentheses '('
    stack_two = [] # stack for '['
    stack_three = [] #stack for '<'
    stack_four = [] #stack for '{'
    pairs = []  # List to store pairs of indices of paired nucleotides
    
    for i, char in enumerate(ipt):
        if char == '(':
            stack_one.append(i)
            
        if char == '[':
            stack_two.append(i)
              
        if char == '<':
            stack_three.append(i)
            
        if char == '{':
            stack_four.append(i)

        #Closing parens
        if char == ')':
            if stack_one:
                pair = (stack_one.pop(), i)
                pairs.append(pair)
                
        if char == ']':
            if stack_two:
                pair = (stack_two.pop(),i)
                pairs.append(pair)
                
        if char == '>':
            if stack_three:
                pair = (stack_three.pop(),i)
                pairs.append(pair)
                
        if char == '}':
            if stack_four:
                pair = (stack_four.pop(),i)
                pairs.append(pair)
    
    return pairs