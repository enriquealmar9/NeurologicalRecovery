#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################
from helper_code import *
import numpy as np, os
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from scipy.signal import butter, lfilter

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.

def train_challenge_model(data_folder, model_folder, verbose):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    batch_size=8
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()
    total_data = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        #try:
        patient_metadata, recording_metadata, recording_data = load_challenge_data2(data_folder, patient_id)
        current_cpc = get_cpc(patient_metadata)
        current_outcome = get_outcome(patient_metadata)
        for i in recording_data:
            # i is a np.array so, we have to apply the preproccesing on it
            i_filter=butter_bandpass_filter(i, 0.5, 30, 100).reshape(18,30000,1)
            total_data.append(i_filter.astype(int))
            cpcs.append(current_cpc)
            outcomes.append(current_outcome)
        #except:
        #    print("Missing file")

    #fetures = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
    cpcs -= 1
    cpcs=to_categorical(cpcs,num_classes=5).tolist()
    outcomes=to_categorical(outcomes,num_classes=2).tolist()
    #total_data = np.asarray(total_data)
    #print(total_data.shape)
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3) 

    total_data = list(divide_chunks(total_data, batch_size))
    outcomes = list(divide_chunks(outcomes, batch_size))
    cpcs = list(divide_chunks(cpcs, batch_size))


    if len(total_data[-1])<batch_size:
        total_data = total_data[ : -1]
        outcomes = outcomes[ : -1]
        cpcs = cpcs[ : -1]

    with tf.device("CPU"):
        train_bi = tf.data.Dataset.from_tensor_slices((total_data,outcomes))#.batch(batch_size)

    model_binary  = EEGNet_binary(nb_classes = 2, Chans = 18, Samples = 30000)
    model_binary.summary()
    model_binary.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer = 'adam',metrics=[keras.metrics.TruePositives()])
    fittedModel = model_binary.fit(train_bi, batch_size=batch_size, epochs=50,callbacks=callback) 
    print("Binary-Outcome model trained and saved")

    with tf.device("CPU"):
        train_multi = tf.data.Dataset.from_tensor_slices((total_data,cpcs))#.batch(batch_size)

    model_multiclass  = EEGNet_multiclass(nb_classes = 5, Chans = 18, Samples = 30000)
    model_multiclass.summary()
    model_multiclass.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=[keras.metrics.TruePositives()])
    #fittedModel = model_multiclass.fit(train_multi, validation_split=0.15, batch_size=batch_size, epochs=50,  callbacks=callback) 
    fittedModel = model_multiclass.fit(train_multi, batch_size=batch_size, epochs=50,  callbacks=callback) 
    # K-fold validation: https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    print("Multiclass-CPC model trained and saved")

    # Save the models.
    save_challenge_model(model_folder, model_binary, model_multiclass)

    if verbose >= 1:
        print('Done.')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    model_binary = keras.models.load_model(os.path.join(model_folder, 'EEGNet_binary.sav'))
    model_multiclass = keras.models.load_model(os.path.join(model_folder, 'EEGNet_multiclass.sav'))
    return [model_binary, model_multiclass]

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    #imputer = models['imputer']
    outcome_model = models[0]
    cpc_model = models[1]

    # Load data.
    cpcs = list()
    total_data = list()
    patient_metadata, recording_metadata, recording_data = load_challenge_data_test(data_folder, patient_id)

    for i in recording_data:
        # i is a np.array so, we have to apply the preproccesing on it
        i_filter=butter_bandpass_filter(i, 0.5, 30, 100)
        total_data.append(i_filter.astype(int))

    total_data = np.array(total_data)
    quality_score = get_quality_scores(recording_metadata)
    quality_score = quality_score[~np.isnan(quality_score)]

    outcome_probability = outcome_model.predict(total_data)
    outcome_probability_mean = get_weighted_label(quality_score, outcome_probability)
    outcome = np.argmax(outcome_probability_mean)

    cpc = cpc_model.predict(total_data)
    cpc_mean = np.argmax(get_weighted_label(quality_score, cpc))+1
    #print("cpc", cpc_mean, np.argmax(cpc_mean))

    #patient_prediction=np.sum(new_predictions, axis=0)/len(new_predictions)
    #print(patient_prediction)
    #cpc=np.mean(patient_prediction.argmax())+1
    #print(cpc)
    #outcome_probability=np.sum(patient_prediction[2:5])
    #print(patient_prediction[2:5])
    #print(outcome_probability)
    #if outcome_probability >= 0.75:
    #    outcome=1
    #elif outcome_probability < 0.75:
    #    outcome=0
    return outcome, outcome_probability_mean[1], cpc_mean

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_weighted_label(quality_scores, probabilities):
    len_x, len_y = probabilities.shape
    weighted_probabilities = np.multiply(probabilities,quality_scores.reshape(len_x,1))
    mean_weighted_probailities = np.sum(weighted_probabilities, axis=0)/np.sum(quality_scores)
    return mean_weighted_probailities

def EEGNet_multiclass(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

def EEGNet_binary(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    sigmoid      = Activation('sigmoid', name = 'sigmoid')(dense)
    
    return Model(inputs=input1, outputs=sigmoid)

def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))  

def load_challenge_data2(data_folder, patient_id):
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)
    recording_metadata = load_text_file(recording_metadata_file)

    # Load recordings.
    recordings = list()
    recording_ids = get_recording_ids(recording_metadata)
    quality_score = get_quality_scores(recording_metadata)

    for i in range(0, len(recording_ids)):
        if recording_ids[i] != 'nan' and quality_score[i]>=0.7:
            recording_location = os.path.join(data_folder, patient_id, recording_ids[i])
            #try:
            recording_data, sampling_frequency, channels = load_recording(recording_location)
            recordings.append(recording_data)
            #except:
            #    print("not considered")
        else:
            recording_data = None
        
    return patient_metadata, recording_metadata, recordings

def load_challenge_data_test(data_folder, patient_id):
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)
    recording_metadata = load_text_file(recording_metadata_file)

    # Load recordings.
    recordings = list()
    recording_ids = get_recording_ids(recording_metadata)

    for i in range(0, len(recording_ids)):
        if recording_ids[i] != 'nan':
            recording_location = os.path.join(data_folder, patient_id, recording_ids[i])
            #try:
            recording_data, sampling_frequency, channels = load_recording(recording_location)
            recordings.append(recording_data)
            #except:
            #    print("not considered")
        else:
            recording_data = None
        
    return patient_metadata, recording_metadata, recordings

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Save your trained model.
def save_challenge_model(model_folder, outcome_model, cpc_model):
    outcome_model.save(os.path.join(model_folder, 'EEGNet_binary.sav'))
    cpc_model.save(os.path.join(model_folder, 'EEGNet_multiclass.sav'))
