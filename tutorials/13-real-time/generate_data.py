# Generate a noise volume given a set of parameters

import os
import glob
import time
import random
import inspect
import typing
import nibabel  # type: ignore
import numpy as np  # type: ignore
from brainiak.utils import fmrisim as sim  # type: ignore
import sys
sys.path.append("..")
import utils

# Template input directory
frame = inspect.currentframe()
moduleFile = typing.cast(str, frame.f_code.co_filename)
moduleDir = os.path.dirname(moduleFile)
fmrisim_dir = os.path.join(moduleDir, "fmrisim/")

# Data output directory
data_dir = os.path.join(utils.results_path, "13-real-time/data/")

# If the folder doesn't exist then make it
if os.path.isdir(data_dir) is False:
    os.makedirs(data_dir, exist_ok=True)

# Specify the volume parameters
trDuration = 2  # seconds
numTRs = 200 # How many TRs will you generate?

# Set up stimulus event time course parameters
event_duration = 10  # How long is each event
isi = 0  # What is the time between each event
burn_in = 0  # How long before the first event

# Specify signal magnitude parameters
signal_change = 10 # How much change is there in intensity for the max of the patterns across participants
multivariate_pattern = 0  # Do you want the signal to be a z scored pattern across voxels (1) or a univariate increase (0)
switch_ROI = 0  # Do you want to switch the ROIs over part way through and if so, specify the proportion of TRs before this happens

print('Load template of average voxel value')
template_nii = nibabel.load(fmrisim_dir + 'sub_template.nii.gz')
template = template_nii.get_data()

dimensions = np.array(template.shape[0:3])

print('Create binary mask and normalize the template range')
mask, template = sim.mask_brain(volume=template,
                                mask_self=True,
                               )

# Write out the mask as a numpy file
np.save(data_dir + 'mask.npy', mask.astype(np.uint8))

# Load the noise dictionary
print('Loading noise parameters')
with open(fmrisim_dir + 'sub_noise_dict.txt', 'r') as f:
    noise_dict = f.read()
noise_dict = eval(noise_dict)
noise_dict['matched'] = 0

print('Generating noise')
noise = sim.generate_noise(dimensions=dimensions,
                           stimfunction_tr=np.zeros((numTRs, 1)),
                           tr_duration=int(trDuration),
                           template=template,
                           mask=mask,
                           noise_dict=noise_dict,
                           )

# Create the stimulus time course of the conditions
total_time = int(numTRs * trDuration)
events = int(total_time / event_duration)
onsets_A = []
onsets_B = []
for event_counter in range(events):
    
    # Flip a coin for each epoch to determine whether it is A or B
    if np.random.randint(0, 2) == 1:
        onsets_A.append(event_counter * event_duration)
    else:
        onsets_B.append(event_counter * event_duration)
        
temporal_res = 0.5 # How many timepoints per second of the stim function are to be generated?

# Create a time course of events 
stimfunc_A = sim.generate_stimfunction(onsets=onsets_A,
                                       event_durations=[event_duration],
                                       total_time=total_time,
                                       temporal_resolution=temporal_res,
                                      )

stimfunc_B = sim.generate_stimfunction(onsets=onsets_B,
                                       event_durations=[event_duration],
                                       total_time=total_time,
                                       temporal_resolution=temporal_res,
                                      )

# Create a labels timecourse
np.save(data_dir + 'labels.npy', (stimfunc_A + (stimfunc_B * 2)))


print('Load ROIs')
nii_A = nibabel.load(fmrisim_dir + 'ROI_A.nii.gz')
nii_B = nibabel.load(fmrisim_dir + 'ROI_B.nii.gz')
ROI_A = nii_A.get_data()
ROI_B = nii_B.get_data()

# How many voxels per ROI
voxels_A = int(ROI_A.sum())
voxels_B = int(ROI_B.sum())

# Create a pattern of activity across the two voxels
print('Creating signal pattern')
if multivariate_pattern == 1:
    pattern_A = np.random.rand(voxels_A).reshape((voxels_A, 1))
    pattern_B = np.random.rand(voxels_B).reshape((voxels_B, 1))
else:  # Just make a univariate increase
    pattern_A = np.tile(1, voxels_A).reshape((voxels_A, 1))
    pattern_B = np.tile(1, voxels_B).reshape((voxels_B, 1))

# Multiply each pattern by each voxel time course
weights_A = np.tile(stimfunc_A, voxels_A) * pattern_A.T
weights_B = np.tile(stimfunc_B, voxels_B) * pattern_B.T

# Convolve the onsets with the HRF
print('Creating signal time course')
signal_func_A = sim.convolve_hrf(stimfunction=weights_A,
                               tr_duration=trDuration,
                               temporal_resolution=temporal_res,
                               scale_function=1,
                               )

signal_func_B = sim.convolve_hrf(stimfunction=weights_B,
                               tr_duration=trDuration,
                               temporal_resolution=temporal_res,
                               scale_function=1,
                               )

# Multiply the signal by the signal change 
signal_func_A *= signal_change
signal_func_B *= signal_change

# Combine the signal time course with the signal volume
print('Creating signal volumes')
signal_A = sim.apply_signal(signal_func_A,
                            ROI_A,
                           )

signal_B = sim.apply_signal(signal_func_B,
                            ROI_B,
                           )

# Do you want to switch the location of the signal 75% of the way through through?
if switch_ROI > 0:
    
    # When does the switch occur?
    switch_point = int(numTRs * switch_ROI)
    
    part_1_A = sim.apply_signal(signal_func_A[:switch_point, :],
                                ROI_A,
                               )
    
    part_2_A = sim.apply_signal(signal_func_A[switch_point:, :],
                                ROI_B,
                               )
    
    part_1_B = sim.apply_signal(signal_func_B[:switch_point, :],
                                ROI_B,
                               )
    
    part_2_B = sim.apply_signal(signal_func_B[switch_point:, :],
                                ROI_A,
                               )
        
    # Concatenate the new volumes    
    signal_A = np.concatenate((part_1_A, part_2_A), axis=3)
    signal_B = np.concatenate((part_1_B, part_2_B), axis=3)
    
#    # What will you name this file as?
#    data_dir = fmrisim_dir + 'data_switched'
    
# Combine the two signal timecourses
signal = signal_A + signal_B

print('Generating TRs in real time')
for idx in range(numTRs):
    
    #  Create the brain volume on this TR
    brain = noise[:, :, :, idx] + signal[:, :, :, idx]
            
    # Save the volume as a numpy file, with each TR as its own file
    output_file = data_dir + 'rt_' + format(idx, '03d') + '.npy'
    
    # Save file 
    brain_float32 = brain.astype(np.float32)
    print("Generate {}".format(output_file))
    np.save(output_file, brain_float32)
    
    # Sleep until next TR
    time.sleep(trDuration)
