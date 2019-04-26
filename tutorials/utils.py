import numpy as np 
import os
import scipy.io
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from copy import deepcopy

# Data path: Where the data for the tutorials is stored.
# Change this path only if you have saved the data to a different folder.
data_path = os.path.join(os.path.expanduser('~'), 'brainiak_datasets')

# Results path: Where the results and intermediate analyses of the tutorials are stored.
# Change this path only if you wish to save your outputs to a different folder.
results_path = os.path.join(os.path.expanduser('~'), 'brainiak_results')

# Data path VDC dataset
vdc_data_dir = os.path.join(data_path, 'vdc') 

# constants for the VDC dataset
vdc_label_dict = {1: "Faces", 2: "Scenes", 3: "Objects"}
vdc_all_ROIs = ['FFA', 'PPA']
vdc_n_runs = 3
vdc_TR = 1.5
vdc_hrf_lag = 4.5  # In seconds what is the lag between a stimulus onset and the peak bold response
vdc_TRs_run = 310

#constants for the simulated data in notebook 02-data-handling
nb2_simulated_data = os.path.join(data_path, '02-data-handling-simulated-dataset')

#constants for ninety six dataset
ns_data_dir = os.path.join(data_path, 'NinetySix')

all_subj_initials = {'BE', 'KO', 'SN', 'TI'}
rois_to_remove = ['lLO', 'rLO']
rois_to_keep = ['lFFA', 'rFFA', 'lPPA', 'rPPA']

#constants for latatt dataset
latatt_dir = os.path.join(data_path, 'latatt')

# constants for the FCMA (face-scene) dataset
fs_data_dir = os.path.join(data_path, 'face_scene')

# for Pieman2 dataset
pieman2_dir =  os.path.join(data_path, 'Pieman2')

# for Raider dataset
raider_data_dir = os.path.join(data_path, 'raider')

# for Sherlock dataset
sherlock_h5_data = os.path.join(data_path, 'sherlock_h5')
sherlock_dir = os.path.join(data_path, 'Sherlock_processed')



def get_MNI152_template(dim_x, dim_y, dim_z):
    """get MNI152 template used in fmrisim
    Parameters
    ----------
    dim_x: int
    dim_y: int
    dim_z: int
        - dims set the size of the volume we want to create
    
    Return
    -------
    MNI_152_template: 3d array (dim_x, dim_y, dim_z)
    """
    # Import the fmrisim from BrainIAK
    import brainiak.utils.fmrisim as sim 
    # Make a grey matter mask into a 3d volume of a given size
    dimensions = np.asarray([dim_x, dim_y, dim_z])
    _, MNI_152_template = sim.mask_brain(dimensions)
    return MNI_152_template


def load_vdc_stim_labels(sub):
    """load the stimulus labels for the VDC data
    Parameters 
    ----------
    sub: string, subject id 
    
    Return
    ----------
    Stimulus labels for all runs 
    """
    stim_label = [];
    stim_label_allruns = [];
    for run in range(1, vdc_n_runs + 1):
        in_file = os.path.join(vdc_data_dir, sub ,'ses-day2','design_matrix','%s_localizer_0%d.mat' % (sub, run))
        # Load in data from matlab
        stim_label = scipy.io.loadmat(in_file);
        stim_label = np.array(stim_label['data']);
        # Store the data
        if run == 1:
            stim_label_allruns = stim_label;
        else:       
            stim_label_allruns = np.hstack((stim_label_allruns, stim_label))
    return stim_label_allruns


def load_vdc_mask(ROI_name, sub):
    """Load the mask for the VDC data 
    Parameters
    ----------
    ROI_name: string
    sub: string 
    
    Return
    ----------
    the requested mask
    """    
    assert ROI_name in vdc_all_ROIs
    maskdir = os.path.join(vdc_data_dir,sub,'preprocessed','masks')
    # load the mask
    maskfile = os.path.join(maskdir, sub + "_ventral_%s_locColl_to_epi1.nii.gz" % (ROI_name))
    mask = nib.load(maskfile)
    print("Loaded %s mask" % (ROI_name))
    return mask


def load_vdc_epi_data(sub, run):
    # Load MRI file (in Nifti format) of one localizer run
    epi_in = os.path.join(vdc_data_dir, sub,
              "preprocessed","loc","%s_filtered2_d1_firstExampleFunc_r%d.nii" % (sub, run))
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data


def mask_data(epi_data, mask): 
    """mask the input data with the input mask 
    Parameters
    ----------
    epi_data
    mask
    
    Return
    ----------
    masked data
    """    
    nifti_masker = NiftiMasker(mask_img=mask)
    epi_masked_data = nifti_masker.fit_transform(epi_data);
    return epi_masked_data


def scale_data(data): 
    data_scaled = preprocessing.StandardScaler().fit_transform(data)
    return data_scaled


# Make a function to load the mask data
def load_vdc_masked_data(directory, subject_name, mask_list):
    masked_data_all = [0] * len(mask_list)

    # Cycle through the masks
    for mask_counter in range(len(mask_list)):
        # load the mask for the corresponding ROI
        mask = load_vdc_mask(mask_list[mask_counter], subject_name)

        # Cycle through the runs
        for run in range(1, vdc_n_runs + 1):
            # load fMRI data 
            epi_data = load_vdc_epi_data(subject_name, run)
            # mask the data 
            epi_masked_data = mask_data(epi_data, mask)
            epi_masked_data = np.transpose(epi_masked_data)
            
            # concatenate data 
            if run == 1:
                masked_data_all[mask_counter] = epi_masked_data
            else:
                masked_data_all[mask_counter] = np.hstack(
                    (masked_data_all[mask_counter], epi_masked_data)
                )
    return masked_data_all



""""""


# Make a function to load the mask data
def load_data(directory, subject_name, mask_name='', num_runs=3, zscore_data=False):
    
    # Cycle through the masks
    print ("Processing Start ...")
    
    # If there is a mask supplied then load it now
    if mask_name is '':
        mask = None
    else:
        mask = load_vdc_mask(mask_name, subject_name)

    # Cycle through the runs
    for run in range(1, num_runs + 1):
        epi_data = load_vdc_epi_data(subject_name, run)
        
        # Mask the data if necessary
        if mask_name is not '':
            epi_mask_data = mask_data(epi_data, mask).T
        else:
            # Do a whole brain mask 
            if run == 1:
                # Compute mask from epi
                mask = compute_epi_mask(epi_data).get_data()  
            else:
                # Get the intersection mask 
                # (set voxels that are within the mask on all runs to 1, set all other voxels to 0)   
                mask *= compute_epi_mask(epi_data).get_data()  
            
            # Reshape all of the data from 4D (X*Y*Z*time) to 2D (voxel*time): not great for memory
            epi_mask_data = epi_data.get_data().reshape(
                mask.shape[0] * mask.shape[1] * mask.shape[2], 
                epi_data.shape[3]
            )

        # Transpose and z-score (standardize) the data  
        if zscore_data == True:
            scaler = preprocessing.StandardScaler().fit(epi_mask_data)
            preprocessed_data = scaler.transform(epi_mask_data)
        else:
            preprocessed_data = epi_mask_data
        
        # Concatenate the data
        if run == 1:
            concatenated_data = preprocessed_data
        else:
            concatenated_data = np.hstack((concatenated_data, preprocessed_data))
    
    # Apply the whole-brain masking: First, reshape the mask from 3D (X*Y*Z) to 1D (voxel). 
    # Second, get indices of non-zero voxels, i.e. voxels inside the mask. 
    # Third, zero out all of the voxels outside of the mask.
    if mask_name is '':
        mask_vector = np.nonzero(mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], ))[0]
        concatenated_data = concatenated_data[mask_vector, :]
        
    # Return the list of mask data
    return concatenated_data, mask


# Make a function for loading in the labels
def load_labels(directory, subject_name):
    stim_label = [];
    stim_label_concatenated = [];
    for run in range(1,4):
        in_file= os.path.join(directory, subject_name, 'ses-day2','design_matrix' ,"%s_localizer_0%d.mat" % (subject_name, run))

        # Load in data from matlab
        stim_label = scipy.io.loadmat(in_file);
        stim_label = np.array(stim_label['data']);

        # Store the data
        if run == 1:
            stim_label_concatenated = stim_label;
        else:       
            stim_label_concatenated = np.hstack((stim_label_concatenated, stim_label))

    print("Loaded ", subject_name)
    return stim_label_concatenated


# Convert the TR
def label2TR(stim_label, num_runs, TR, TRs_run):

    # Calculate the number of events/run
    _, events = stim_label.shape
    events_run = int(events / num_runs)    
    
    # Preset the array with zeros
    stim_label_TR = np.zeros((TRs_run * 3, 1))

    # Cycle through the runs
    for run in range(0, num_runs):

        # Cycle through each element in a run
        for i in range(events_run):

            # What element in the concatenated timing file are we accessing
            time_idx = run * (events_run) + i

            # What is the time stamp
            time = stim_label[2, time_idx]

            # What TR does this timepoint refer to?
            TR_idx = int(time / TR) + (run * (TRs_run - 1))

            # Add the condition label to this timepoint
            stim_label_TR[TR_idx]=stim_label[0, time_idx]
        
    return stim_label_TR

# Create a function to shift the size
def shift_timing(label_TR, TR_shift_size):
    
    # Create a short vector of extra zeros
    zero_shift = np.zeros((TR_shift_size, 1))

    # Zero pad the column from the top.
    label_TR_shifted = np.vstack((zero_shift, label_TR))

    # Don't include the last rows that have been shifted out of the time line.
    label_TR_shifted = label_TR_shifted[0:label_TR.shape[0],0]
    
    return label_TR_shifted


# Extract bold data for non-zero labels.
def reshape_data(label_TR_shifted, masked_data_all):
    label_index = np.nonzero(label_TR_shifted)
    label_index = np.squeeze(label_index)
    
    # Pull out the indexes
    indexed_data = np.transpose(masked_data_all[:,label_index])
    nonzero_labels = label_TR_shifted[label_index] 
    
    return indexed_data, nonzero_labels

# Take in a brain volume and label vector that is the length of the event number and convert it into a list the length of the block number
def blockwise_sampling(eventwise_data, eventwise_labels, eventwise_run_ids, events_per_block=10):
    
    # How many events are expected
    expected_blocks = int(eventwise_data.shape[0] / events_per_block)
    
    # Average the BOLD data for each block of trials into blockwise_data
    blockwise_data = np.zeros((expected_blocks, eventwise_data.shape[1]))
    blockwise_labels = np.zeros(expected_blocks)
    blockwise_run_ids = np.zeros(expected_blocks)
    
    for i in range(0, expected_blocks):
        start_row = i * events_per_block 
        end_row = start_row + events_per_block - 1 
        
        blockwise_data[i,:] = np.mean(eventwise_data[start_row:end_row,:], axis = 0)
        blockwise_labels[i] = np.mean(eventwise_labels[start_row:end_row])
        blockwise_run_ids[i] = np.mean(eventwise_run_ids[start_row:end_row])
        
    # Report the new variable sizes
    print('Expected blocks: %d; Resampled blocks: %d' % (expected_blocks, blockwise_data.shape[0]))

    # Return the variables downsampled_data and downsampled_labels
    return blockwise_data, blockwise_labels, blockwise_run_ids




def normalize(bold_data_, run_ids):
    """normalized the data within each run
    
    Parameters
    --------------
    bold_data_: np.array, n_stimuli x n_voxels
    run_ids: np.array or a list
    
    Return
    --------------
    normalized_data
    """
    scaler = StandardScaler()
    data = []
    for r in range(vdc_n_runs):
        data.append(scaler.fit_transform(bold_data_[run_ids == r, :]))
    normalized_data = np.vstack(data)
    return normalized_data
    
    
def decode(X, y, cv_ids, model): 
    """
    Parameters
    --------------
    X: np.array, n_stimuli x n_voxels
    y: np.array, n_stimuli, 
    cv_ids: np.array - n_stimuli, 
    
    Return
    --------------
    models, scores
    """
    scores = []
    models = []
    ps = PredefinedSplit(cv_ids)
    for train_index, test_index in ps.split():
        # split the data 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit the model on the training set 
        model.fit(X_train, y_train)
        # calculate the accuracy for the hold out run
        score = model.score(X_test, y_test)
        # save stuff 
        models.append(deepcopy(model))
        scores.append(score)
    return models, scores

"""helper funcs
"""

def load_data_for_a_subj(subj_initials):
    assert subj_initials in all_subj_initials
    images = scipy.io.loadmat(
        os.path.join(ns_data_dir, '%s_images.mat' % (subj_initials))
    )['images']  
    data = scipy.io.loadmat(
        os.path.join(ns_data_dir, '%s_roi_data.mat' % (subj_initials))
    ) 
    
    # Unpack metadata 
    roi_data_all = data['roi_data']
    roi_names = data['roinames']
    labels = np.array(data['labels'])
    categoryNames = data['categoryNames']

    # Re-format metadata labels and ROIs
    n_categories = categoryNames.shape[1]
    n_rois = roi_names.shape[1]
    categories = [categoryNames[0, i][0] for i in range(n_categories)]
    roi_names = [roi_names[0, i][0] for i in range(n_rois)]
    labels = np.squeeze(labels) 
    label_dict = {categories[i]: i+1 for i in range(len(categories))}

    # Remove r/lLO
    roi_data = []
    for r in range(n_rois): 
        if roi_names[r] in rois_to_keep: 
            roi_data.append(roi_data_all[0, r])
    roi_names = rois_to_keep
    n_rois = len(rois_to_keep)
    return images, roi_data, roi_names, n_rois, categories, n_categories, labels, label_dict


def digitize_rdm(rdm_raw, n_bins = 10): 
    """Digitize an input matrix to n bins (10 bins by default)
    rdm_raw: a square matrix 
    """
    # compute the bins 
    
    rdm_bins = [np.percentile(np.ravel(rdm_raw), 100/n_bins * i) for i in range(n_bins)]
    # Compute the vectorized digitized value 
    rdm_vec_digitized = np.digitize(np.ravel(rdm_raw), bins = rdm_bins) * (100 // n_bins) 
    
    # Reshape to matrix
    rdm_digitized = np.reshape(rdm_vec_digitized, np.shape(rdm_raw)) 
    
    # Force symmetry in the plot
    rdm_digitized = (rdm_digitized + rdm_digitized.T) / 2
    
    return rdm_digitized
