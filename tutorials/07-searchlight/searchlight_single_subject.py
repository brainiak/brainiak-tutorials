# Run a whole brain searchlight on a single subject in the VDC dataset

# Import libraries
import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import LinearSVC
from scipy.stats import zscore
import os, sys

# Import additional libraries you need
sys.path.append('../')
# load some helper functions
from utils import load_vdc_mask, load_vdc_epi_data, load_vdc_stim_labels, label2TR, shift_timing
# load some constants
from utils import vdc_data_dir, results_path,vdc_all_ROIs, vdc_label_dict, vdc_n_runs, vdc_hrf_lag, vdc_TR, vdc_TRs_run

# parameters
sub = 'sub-01'
roi_name = 'FFA'

# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Output data Path
output_path = os.path.join(results_path,'searchlight_results')
if rank == 0:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
# load mask of the subject
mask = load_vdc_mask(roi_name, sub)
mask = mask.get_data()
coords = np.where(mask)

# load labels of the subject in all ranks
stim_label_allruns = load_vdc_stim_labels(sub)
stim_label_TR = label2TR(stim_label_allruns, vdc_n_runs, vdc_TR, vdc_TRs_run)
shift_size = int(vdc_hrf_lag / vdc_TR)
label = shift_timing(stim_label_TR, shift_size)
# extract non-zero labels
label_index = np.squeeze(np.nonzero(label))
# Pull out the indexes
labels = label[label_index] 

# get run ids (works similarity to cv_ids)
run_ids = stim_label_allruns[5,:] - 1
# split data according to run ids
ps = PredefinedSplit(run_ids)

# Same them as the broadcast variables
bcvar = [labels, ps]

# load the data in rank 0
if rank == 0:
    # Make a function to load the data
    def load_data(directory, subject_name):
        # Cycle through the runs
        for run in range(1, vdc_n_runs + 1):
            epi_data = load_vdc_epi_data(subject_name, run)
            bold_data = epi_data.get_data()
            affine_mat = epi_data.affine
            dimsize = epi_data.header.get_zooms()
            # Concatenate the data
            if run == 1:
                concatenated_data = bold_data
            else:
                concatenated_data = np.concatenate((concatenated_data, bold_data), axis=-1)
        return concatenated_data, affine_mat, dimsize

    data, affine_mat, dimsize = load_data(vdc_data_dir, sub)
    # extract bold data for non-zero labels
    data = data[:, :, :, label_index]
    # normalize the data within each run
    for r in range(vdc_n_runs):
        data[:, :, :, run_ids==r] = np.nan_to_num(zscore(data[:, :, :, run_ids==r], axis=3))
else:
    data = None

# Set parameters
sl_rad = 1
max_blk_edge = 5
pool_size = 1

# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)

# Distribute the information to the searchlights (preparing it to run)
sl.distribute([data], mask)

# Broadcast variables
sl.broadcast(bcvar)

# Set up the kernel function, in this case an SVM
def calc_svm(data, sl_mask, myrad, bcvar):
    if np.sum(sl_mask) < 14:
        return -1
    scores = []
    labels, ps = bcvar[0], bcvar[1]

    # Reshape the data
    sl_num_vx = sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2]
    num_epoch = data[0].shape[3]
    data_sl = data[0].reshape(sl_num_vx, num_epoch).T

    # Classifier: loop over all runs to leave each run out once
    model = LinearSVC()
    for train_index, test_index in ps.split():
        X_train, X_test = data_sl[train_index], data_sl[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # Fit a svm
        model.fit(X_train, y_train)
        # Calculate the accuracy for the hold out run
        score = model.score(X_test, y_test)
        scores.append(score)
        
    return np.mean(scores)

# Run the searchlight analysis
print("Begin SearchLight in rank %s\n" % rank)
sl_result = sl.run_searchlight(calc_svm, pool_size=pool_size)
print("End SearchLight in rank %s\n" % rank)

# Only save the data if this is the first core
if rank == 0:
    # Convert NaN to 0 in the output
    sl_result = np.nan_to_num(sl_result[mask==1])
    # Reshape
    result_vol = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))  
    result_vol[coords[0], coords[1], coords[2]] = sl_result   
    # Convert the output into what can be used
    result_vol = result_vol.astype('double')   
    # Save the average result
    output_name = os.path.join(output_path, '%s_%s_SL.nii.gz' % (sub, roi_name))
    sl_nii = nib.Nifti1Image(result_vol, affine_mat)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nib.save(sl_nii, output_name)  # Save    
    
    print('Finished searchlight')
