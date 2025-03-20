"""
Module for creating/training GARDN architectures

Created on Mon Jul 24 14:01:33 2022


@author: aidan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nupack as n
import time
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

from tfa_spectral import SpectralNormalization as SN

import GA_util
import util
import agreement
from attention_util import SelfAttnModel



############### GARDN Model Definitions ##############################

#-------------------------------------------------------------------
def create_GARDN_toehold_generator(sequence_length=60,
                                   latent_dim=128,
                                   filters=64,
                                   act='relu',
                                   kernel_1 = (4,9),
                                   kernel_2 = (2,5),
                                   kernel_3 = (1,3)):
                                   

    input_layers = tfkl.Input((latent_dim,))
    x = tfkl.Reshape((1, 1, latent_dim))(input_layers)


    x = tfkl.UpSampling2D(size=(2,5))(x)
    x = SN(tfkl.Conv2D(filters=filters,kernel_size=kernel_1,strides=(1,1),padding='same',activation=act))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.UpSampling2D(size=(2,6))(x)
    x = SN(tfkl.Conv2D(filters=filters//2,kernel_size=kernel_2,strides=(1,1),padding='same',activation=act))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.UpSampling2D(size=(1,2))(x)
    x = SN(tfkl.Conv2D(filters=filters//4,kernel_size=kernel_3,strides=(1,1),padding='same',activation=act))(x)
    x = tfkl.BatchNormalization()(x)

    #Reverse Complement
    FIVE_PRIME = x[:,:,:13,:]
    STEM_1 = x[:,:,13:31,:]
    RC = tf.reverse(STEM_1,(2,1)) 
    x = tf.concat((FIVE_PRIME,STEM_1,x[:,:,31:42,:],RC[:,:,:6,:],x[:,:,48:51,:],RC[:,:,9:,:]),axis=2)  

    x = SN(tfkl.Conv2D(filters=1, kernel_size=(1,1),strides=(1,1),padding='same',activation='tanh'))(x)
    x = tfkl.BatchNormalization()(x)




    generator  = tfk.Model(input_layers, x)
    
    return generator


#-------------------------------------------------------------------
def create_GARDN_toehold_discriminator(sequence_length=60,
                                     latent_dim=128,
                                     act = tfkl.LeakyReLU(0.2),
                                     kernel_1 = (4,18),
                                     kernel_2 = (1,9),
                                     kernel_3 = (1,3)
                                    ):
    
    input_seqs = tfk.Input(shape=(4,sequence_length,1))


    y = SN(tfkl.Conv2D(latent_dim/2, kernel_1, strides=(4, 1), padding="same",activation=act))(input_seqs)
    y = tfkl.LayerNormalization()(y)
    y = SN(tfkl.Conv2D(latent_dim/4, kernel_2, strides=(4, 1), padding="same",activation=act))(y)
    y = tfkl.LayerNormalization()(y)

    y = SN(tfkl.Conv2D(latent_dim/8,kernel_3,strides=(4,1),padding='same',activation=act))(y)
    y = tfkl.LayerNormalization()(y)

    y = tfkl.Flatten()(y)

    z = SN(tfkl.Dense(1,activation=None))(y)



    discriminator = tfk.Model(inputs=input_seqs,outputs=z,name='discriminator')
    return discriminator



#-------------------------------------------------------------------
def create_GARDN_utr_generator(sequence_length=50,latent_dim=128,filters=96):
    # latent_dim = 128
    # sequence_length = 50
    # filters = 96

    input_layers = tfkl.Input((latent_dim,))
    x = tfkl.Reshape((1, 1, latent_dim))(input_layers)
    x = SN(tfkl.Conv2DTranspose(filters=filters,kernel_size=(4,9),strides=(2,5),padding='same'))(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    
    x = SN(tfkl.Conv2DTranspose(filters=filters//2,kernel_size=(2,5),strides=(2,5),padding='same'))(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    
    x = SN(tfkl.Conv2DTranspose(filters=filters//4,kernel_size=(1,3),strides=(1,2),padding='same'))(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    
    x, attn1 = SelfAttnModel(filters//4)(x)
    x = SN(tfkl.Conv2DTranspose(filters=1, kernel_size=(1,1),strides=(1,1),padding='same',activation='tanh'))(x)
    generator = tfk.models.Model(input_layers, x)

    return generator



# latent_dim = 128
# input_seqs = keras.Input(shape=(4,50,1))






#-------------------------------------------------------------------
def create_GARDN_utr_discriminator(sequence_length=50,latent_dim=128):

    input_seqs = tfk.Input(shape=(4,sequence_length,1))
    y = SN(tfkl.Conv2D(latent_dim/2, (4, 18), strides=(4, 1), padding="same",activation=tfkl.LeakyReLU(alpha=0.2)))(input_seqs)
    y = tfkl.LayerNormalization()(y)
    
    y = SN(tfkl.Conv2D(latent_dim/4, (1, 9), strides=(4, 1), padding="same",activation=tfkl.LeakyReLU(alpha=0.2)))(y)
    y = tfkl.LayerNormalization()(y)
    
    # y,attn2 = SelfAttnModel(latent_dim/4)(y)
    y = SN(tfkl.Conv2D(latent_dim/8,(1,3),strides=(4,1),padding='same',activation=tfkl.LeakyReLU(alpha=0.2)))(y)
    y = tfkl.LayerNormalization()(y)    
    y = tfkl.Flatten()(y)
    
    z = SN(tfkl.Dense(1,activation=None))(y)
    discriminator = tfk.Model(inputs=input_seqs,outputs=z,name='discriminator_UTR')
    
    return discriminator



############### Training Functions ##############################

#-------------------------------------------------------------------
@tf.function
def w_loss(y_true, y_pred):
    # print(y_true)
    return tf.reduce_mean(y_true * y_pred)


#-------------------------------------------------------------------
@tf.function
def gradient_penalty(g,d,real,fake,gp_lambda=10):

    alpha = np.random.uniform(size=[len(real),1,1,1], low=0., high=1.)

    interpolated = alpha * real + (1 - alpha) * fake
    
    with tf.GradientTape() as tape_p:
        tape_p.watch(interpolated)
        logit = d(interpolated)
        
    grad = tape_p.gradient(logit, interpolated)
    grad_norm = tf.norm(tf.reshape(grad, (real.shape[0], -1)), axis=1)

    return gp_lambda * tf.reduce_mean(tf.square(grad_norm - 1.))

#-------------------------------------------------------------------
def plot_model_structure_quality(generator_model,n_switches=1000,latent_dim=80,my_model=None):
    #inputs should be a trained generator model and the nubmer of switches to generate to evaluate 2nd structure
    
   
    
    random_latent_vectors = np.random.normal(size=(n_switches, latent_dim))
    
    # Decode them to fake switches
    generated_switches = generator_model(random_latent_vectors)
    
    

    dist_save = []
    for i in range(n_switches):
        dist_save.append(GA_util.calc_struc_dist(generated_switches[i,:,:,0],my_model))
    print('Mean:',np.mean(dist_save))
    print('Variance:',np.var(dist_save))
    plt.figure()
    plt.title('Generator Structure Distance')
    plt.scatter(np.arange(1,len(dist_save)+1),dist_save)
    plt.show()
    
    return generated_switches,dist_save


#The next two functions get much better performance with tf.function decorators but become unstable after multiple calls

#-------------------------------------------------------------------
@tf.function
def train_discriminator_step(g,d,real_sequence, noise_z,d_opt,gp_lambda=10):
    with tf.GradientTape() as tape_d:
        
        fake_sequence = g(noise_z, training=False)
        # fake_sequence = tf.expand_dims(tf.transpose(fake_sequence,(0,2,1)),axis=-1)
        # fake_sequence = tf.expand_dims(fake_sequence,axis=-1)


        real_pred = d(real_sequence, training=True)
        fake_pred = d(fake_sequence, training=True)

        y_true = tf.ones(shape=tf.shape(real_pred), dtype=tf.float32)
        real_loss = w_loss(-y_true, real_pred)
        fake_loss = w_loss(y_true, fake_pred)

        gp = gradient_penalty(g,d,real_sequence, fake_sequence,gp_lambda)
            
        total_loss = real_loss + fake_loss + gp

    gradients = tape_d.gradient(total_loss, d.trainable_variables)
    d_opt.apply_gradients(zip(gradients, d.trainable_variables))
        
    return total_loss, gp

#-------------------------------------------------------------------
@tf.function
def train_generator_step(g,d,noise_z,g_opt):
    with tf.GradientTape() as tape_g:
        
        fake_sequence  = g(noise_z, training=True)
        # fake_sequence = tf.expand_dims(fake_sequence,axis=-1)
        # fake_sequence = tf.expand_dims(tf.transpose(fake_sequence,(0,2,1)),axis=-1)
        

        fake_pred = d(fake_sequence, training=False)

        g_loss = w_loss(fake_pred, -tf.ones(shape=tf.shape(fake_pred), dtype=tf.float32))

        gradients = tape_g.gradient(g_loss, g.trainable_variables)
        g_opt.apply_gradients(zip(gradients, g.trainable_variables))
        
    return g_loss


#-------------------------------------------------------------------
def train(g,d,sequences,epochs,discriminator_steps=3,batch_size=64,g_lr=0.0004,d_lr=0.0001,z_dim=128,g_loss_save=None,d_loss_save=None,gp_loss_save=None,score_structure=False,ipt_struc='.............(((((((((...((((((...........))))))...)))))))))'):
    
    
    if d_loss_save is None:
        d_loss_save = []
    if g_loss_save is None:
        g_loss_save = []
    if gp_loss_save is None:
        gp_loss_save = []

    
    mod = n.Model()
    
    g_opt = tfk.optimizers.legacy.Adam(learning_rate=g_lr,beta_1=0.5,beta_2=0.9)
    d_opt = tfk.optimizers.legacy.Adam(learning_rate=d_lr,beta_1=0.5,beta_2=0.9)
    
    dataset_sequences = sequences.shuffle(buffer_size=1024).batch(batch_size)
    
       
    for epoch in range(epochs):
        
        print('epoch: {}'.format(epoch))
        
        epoch_start_time = time.time()
        
        for step,real_sequences in enumerate(dataset_sequences):
            
            
            z = np.random.normal(size=(real_sequences.shape[0], z_dim))
            
            for i in range(discriminator_steps):
                d_loss, gp_loss = train_discriminator_step(g,d,real_sequences,z,d_opt)
            
            g_loss = train_generator_step(g,d,z,g_opt)
            
            

            if step % 200 == 0: #Printing Sequences and 2nd Structure of Generated Constructs
                print('Step %s'%(step))
                print('d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}'.format(d_loss.numpy(), gp_loss.numpy(), g_loss.numpy()))
                tmp = np.random.normal(size=(10,z_dim))
                tmp = g(tmp)
                print(util.unencode(tmp[0,:,:,0]))
                if score_structure:
                    val = agreement.score_structure_against_consensus(tmp[:,:,:,0],ipt_struc)
                    print('Agreement: %.2f'%val)


            
            

        
        # GA_util.plot_model_structure_quality(g,n_switches=20,latent_dim=z_dim,my_model=mod)
        print('epoch {}/{} ({:.2f} sec):, d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}'.format(
            epoch+1, epochs,
            time.time() - epoch_start_time,
            d_loss.numpy(), gp_loss.numpy(), g_loss.numpy()))


        
        
        if epoch % 1 == 0:
            a = np.random.normal(size=(100,z_dim))
            a = g(a)
            util.plot_logo(real_sequences[:,:,:,0])
            util.plot_logo(a[:,:,:,0])
            plt.show()
        #     plot_model_structure_quality(g,n_switches=20,latent_dim=z_dim,my_model=mod)
        
        d_loss_save.append(d_loss)
        g_loss_save.append(g_loss)
        gp_loss_save.append(gp_loss)
        
    return g_loss_save,d_loss_save,gp_loss_save


#-------------------------------------------------------------------
#Track structural agreement over model training iterations
def track_agreement(g,d,sequences,epochs,discriminator_steps=3,batch_size=64,g_lr=0.0004,d_lr=0.0001,z_dim=128,g_loss_save=None,d_loss_save=None,gp_loss_save=None):
    
    
    if d_loss_save is None:
        d_loss_save = []
    if g_loss_save is None:
        g_loss_save = []
    if gp_loss_save is None:
        gp_loss_save = []

    agreement_save = []
    mod = n.Model()
    
    g_opt = tfk.optimizers.legacy.Adam(learning_rate=g_lr,beta_1=0.5,beta_2=0.9)
    d_opt = tfk.optimizers.legacy.Adam(learning_rate=d_lr,beta_1=0.5,beta_2=0.9)
    
    dataset_sequences = sequences.shuffle(buffer_size=1024).batch(batch_size)
    
       
    for epoch in range(epochs):
        
        print('epoch: {}'.format(epoch))
        
        epoch_start_time = time.time()
        
        for step,real_sequences in enumerate(dataset_sequences):
            
            
            z = np.random.normal(size=(real_sequences.shape[0], z_dim))
            
            for i in range(discriminator_steps):
                d_loss, gp_loss = train_discriminator_step(g,d,real_sequences,z,d_opt)
            
            g_loss = train_generator_step(g,d,z,g_opt)
            
            if step % 1 == 0: #Printing Sequences and 2nd Structure of Generated Constructs
                print('Step %s'%(step))
                print('d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}'.format(d_loss.numpy(), gp_loss.numpy(), g_loss.numpy()))
                tmp = np.random.normal(size=(10,z_dim))
                tmp = g(tmp)
                val = agreement.score_structure_against_consensus(tmp[:,:,:,0],'.............(((((((((...((((((...........))))))...)))))))))')
                agreement_save.append(val)
                print('Agreement: %.2f'%val)


            
            

        
        # GA_util.plot_model_structure_quality(g,n_switches=20,latent_dim=z_dim,my_model=mod)
        print('epoch {}/{} ({:.2f} sec):, d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}'.format(
            epoch+1, epochs,
            time.time() - epoch_start_time,
            d_loss.numpy(), gp_loss.numpy(), g_loss.numpy()))


        
        
        if epoch % 1 == 0:
            a = np.random.normal(size=(100,z_dim))
            a = generator(a)
            util.plot_logo(real_sequences[:,:,:,0])
            util.plot_logo(a[:,:,:,0])
            plt.show()
        #     plot_model_structure_quality(g,n_switches=20,latent_dim=z_dim,my_model=mod)
        
        d_loss_save.append(d_loss)
        g_loss_save.append(g_loss)
        gp_loss_save.append(gp_loss)
        
    return g_loss_save,d_loss_save,gp_loss_save,agreement_save



def create_GARDN_RBS_generator(sequence_length=17,latent_dim=64,filters=128):
    
    # latent_dim = 128
    # sequence_length = 50
    # filters = 160

    input_layers = tfkl.Input((latent_dim,))
    x = tfkl.Reshape((1, 1, latent_dim))(input_layers)
    x = SN(tfkl.Conv2DTranspose(filters=filters,kernel_size=(4,9),strides=(2,3),padding='same'))(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    x = SN(tfkl.Conv2DTranspose(filters=filters//2,kernel_size=(2,5),strides=(2,3),padding='same'))(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    x = SN(tfkl.Conv2DTranspose(filters=filters//4,kernel_size=(1,3),strides=(1,2),padding='same'))(x)
    x = x[:,:,:17,:]
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    x, attn1 = SelfAttnModel(filters//4)(x)
    x = SN(tfkl.Conv2DTranspose(filters=1, kernel_size=(1,1),strides=(1,1),padding='same',activation='hard_sigmoid'))(x)
    
    

    generator = tfk.models.Model(input_layers, x)
    


    return generator


def create_GARDN_RBS_discriminator(sequence_length=17,filters=128,act = tfkl.LeakyReLU(0.2),kernel_1 = (4,50),kernel_2 = (2,9),kernel_3 = (1,3), kernel_4 =(1,3)):
    
    input_seqs = tfk.Input(shape=(4,sequence_length,1))

    y = SN(tfkl.Conv2D(filters, kernel_1, strides=(4, 1), padding="same",activation=act))(input_seqs)
    y = tfkl.LayerNormalization()(y)
    
    y = SN(tfkl.Conv2D(filters//2, kernel_2, strides=(4, 1), padding="same",activation=act))(y)
    y = tfkl.LayerNormalization()(y)
    
    y = SN(tfkl.Conv2D(filters//4, kernel_3, strides=(4, 1), padding="same",activation=act))(y)
    y = tfkl.LayerNormalization()(y)
    
    # adding an extra layer
    y = SN(tfkl.Conv2D(filters//8, kernel_4, strides=(4, 1), padding="same", activation=act))(y)
    y = tfkl.LayerNormalization()(y)
    

    y = tfkl.Flatten()(y)

    z = SN(tfkl.Dense(1,activation=None))(y)



    discriminator = tfk.Model(inputs=input_seqs,outputs=z,name='discriminator')
    return discriminator