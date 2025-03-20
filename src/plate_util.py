"""Utils for loading and processing plate reader data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

#Path initialization
concentration_path = 'data/concentration_read_outs'

plate_reader_path = 'data/plate_reader'


#Some settings for how each experimental plate is set up
#These settings could change for a different (e.g 96 well) plate setup or a setup where ONs and OFFs are located in different spots

letters_top = ['A','B','C','D','E','F','G','H']
letters_bottom = ['I','J','K','L','M','N','O','P']

ON_half_X = ['%s%s'%(j,i) for j in letters_top for i in range(1,13,2)]
OFF_half_X = ['%s%s'%(j,i) for j in letters_top for i in range(2,13,2)]
ON_quarter_X =['%s%s'%(j,i) for j in letters_top for i in range(13,25,2)]
OFF_quarter_X = ['%s%s'%(j,i) for j in letters_top for i in range(14,25,2)]
ON_one_X = ['%s%s'%(j,i) for j in letters_bottom for i in range(1,13,2)]
OFF_one_X= ['%s%s'%(j,i) for j in letters_bottom for i in range(13,25,2)]

#Control locations
POS_CTRL = ['I24','J24','K24']
NEG_CTRL = ['L24','M24','N24']
BLK = ['I22','J22','K22','L22']

#Settings for extracting out the ordered set of od260 measurements
valid_wells_od260 = ['2','3','5','6','8','9']
od260 = ['%s%s'%(i,j) for i in letters_top for j in valid_wells_od260]

#------------------------------------------------
def ext_coeff_RNA(seq):

    stack_table = np.array([[27.4,21,25,24],
    [21,14.2,17.8,16.2],
    [25.2,17.4,21.6,21.2],
    [24.6,17.2,20,19.6]]) * 1000
    
    single_table = np.array([15.4,7.2,11.5,9.9])*1000

    base_order = ['A','C','G','U']
    
    

    seq = seq.replace('T','U')
    seq_indices = np.zeros(len(seq),dtype='int');
    for i in range(len(seq)):
        seq_indices[i] = base_order.index(seq[i])
    ext_coeff = 0
    for j in range(1,len(seq)-1):
        # print(j)
        # print(seq_indices[j])
        ext_coeff = ext_coeff - single_table[seq_indices[j]];

    for k in range(1,len(seq)):
        idx_1 = seq_indices[k-1]
        idx_2 = seq_indices[k]
        ext_coeff = ext_coeff + stack_table[idx_1,idx_2]
        
    return ext_coeff


#####OD260 Helpers
#------------------------------------------------
def plate_reader_file_name_to_od260_file_name(plate_reader_file_name,concentration_data_dir):

    
    concentration_file_name = ''
    multiply_4 = False
    end = 'FINAL.xlsx'

    split = plate_reader_file_name.split('_')
    # print(split)
    
    row_start_plate_reader = split[-3]
    row_start_plate_reader = row_start_plate_reader[0].upper() + row_start_plate_reader[1:]
    row_stop_plate_reader = split[-2]
    row_stop_plate_reader = row_stop_plate_reader[0].upper() + row_stop_plate_reader[1:]
    
    numbers_plate_reader = split[-1].strip('.xlsx').split('-')

    number_start_plate_reader = numbers_plate_reader[0]
    number_stop_plate_reader = numbers_plate_reader[1]
    

    
    for file in os.listdir(concentration_path):
        if file[0] != '2':
            pass
        else:
            if file.split('_')[-1] != end:
                pass
            else:
                rows_concentration = file.split('_')[3].split('-')

                row_start_concentration = rows_concentration[0]
                row_stop_concentration = rows_concentration[1]


                numbers_concentration = file.split('_')[4].split('-')
                
                number_start_concentration = numbers_concentration[0]
                number_stop_concentration = numbers_concentration[1]
                
                
                if number_start_plate_reader == number_start_concentration and row_start_plate_reader == row_start_concentration:
                    
                    concentration_file_name = file
                    

                
                    
                    if '1-4' in file.split('_'):
                        
                        multiply_4 = True

    
    
    
    return concentration_file_name,multiply_4

#------------------------------------------------
def load_od260_data(path):
    

    tmp = pd.read_excel(path)
    cols = tmp.iloc[1,:]
    # print(cols.values)
    tmp_2 = tmp.iloc[2:]
    
    cols = [str(i) for i in cols.values]
    
    cols[-4] = '260' #hard coding to fix pandas sometimes laoding this as a string,int, or float

    tmp_2.columns = cols
    
    tmp_2 = tmp_2.replace('?????', np.nan)
    
    return tmp_2



#------------------------------------------------
def extract_sorted_od260_values(loaded_od260_file):

    max_val= int(np.max(loaded_od260_file['Sample Read#']))
    
    
    master_measurements = np.zeros((max_val,48))
    
    #Somtimes the 260 col is loaded as 260 sometimes as 260.0

    for k in range(1,max_val):

        sample_read = k

        this_measurement = loaded_od260_file[loaded_od260_file['Sample Read#'] == sample_read]


        # this_measurement = this_measurement.sort_values('Location') #This is to map them 1-1 with the plate reader data

        master_measurements[k-1,:] = this_measurement['260'].values
        
    return np.nanmean(master_measurements,axis=0)



####Plate Reader helpers
#------------------------------------------------
def process_plate_data(path,save_name = None):
    """function to process the plate reader data into an ammenable format"""
    
    df = pd.read_excel(path)

    df_2 = df.iloc[51:,:]
    
    # df_2 = df_2.reset_index()
    
    # print(df_2)
    #Step 1 is to find all the locations where a column starts
    start_save = []

    for i in range(df_2.shape[0]):
        # print(i)

        if df_2.iloc[i,1] == 'Time':
            start_save.append(i)
            
    #Step 2 is to finad all the stop locations where a column ends
    stop_save = [start_save[1]]
    for i in range(1,len(start_save)):
        stop_idx = start_save[i] + (start_save[i]-start_save[i-1])
        stop_save.append(stop_idx)
        
        
    #Step 3 is to merge each of the sections together, 4 total
    piece_one = df_2.iloc[start_save[0]:stop_save[0],:]


    piece_one_cols = piece_one.iloc[0,:]

    piece_one = piece_one.iloc[1:,:]

    piece_one.columns = piece_one_cols
    piece_one = piece_one.reset_index()




    for j in range(1,int(len(start_save)/2)):
        # print(j)

        piece_next = df_2.iloc[start_save[j]:stop_save[j],:]

        piece_next_cols = piece_next.iloc[0,:]

        piece_next = piece_next.iloc[1:,:]

        piece_next.columns = piece_next_cols

        piece_next = piece_next.reset_index()

        piece_one = pd.concat([piece_one,piece_next],axis=1)
    
    
    piece_one = piece_one.iloc[:-3,:]
    piece_one = piece_one.loc[:, ~piece_one.columns.duplicated()]
    if save_name is None:
        pass
    else:
        piece_one.to_csv('%s.csv'%save_name)
    
    return piece_one


#------------------------------------------------
def get_master_locations_from_file_name(file_name):
    
    """This will return a list of which sequences in the DNA-ordered plate are contained within the plate reader file named file_name"""
    split = file_name.split('_')
    
    letter_start = split[-3][-1]
    
    letter_stop = split[-2][-1]
    
    if letter_start in letters_top:
        letters_for_this_file = letters_top
        
    else:
        letters_for_this_file = letters_bottom
        
    
         
    
    split_split = split[-1].split('-')
    
    start_number = int(split_split[0])
    stop_number = int(split_split[-1][:2])
    
    
    output = ['%s%s'%(letter,number) for letter in letters_for_this_file for number in range(start_number,stop_number+1,2)]
    
    return output

#------------------------------------------------
def get_data_at_time(target_time,df):

    time_obj = datetime.strptime(target_time, '%H:%M:%S')

    return df[df['Time'] == time_obj.time()]

#------------------------------------------------
def get_data_at_time_range(loaded_plate_reader_data,target_time='3:00:00',plus_minus = 50):
    """plus_minus is how long of a time window to average over, by time samples (depends on frequency of collection)"""
    
    time_lookup_val = datetime.strptime(target_time, "%H:%M:%S").time()
    print(time_lookup_val)
    
    out_cols = loaded_plate_reader_data.columns[4:] #Doing this so values are still indexable by well r
    for i,time in enumerate(loaded_plate_reader_data['Time']):
        
        if time > time_lookup_val:
            identified_idx = i
            break
    
    data_sections = loaded_plate_reader_data.iloc[identified_idx-plus_minus:identified_idx+plus_minus,4:]
    
    return data_sections.mean(axis=0)