# Run a whole brain searchlight

# Import libraries
import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from scipy.spatial.distance import euclidean
import os
import pickle 

# Import additional libraries you need
fs_data_dir = os.path.expanduser('~/searchlight_data')

num_subj = 3

# Load and perpare data for one subject
def load_fs_data(sub_id, mask=''):
    # find file path
    sub = 'sub-%.2d' % (sub_id)
    input_dir = os.path.join(fs_data_dir, sub)
    data_file = os.path.join(input_dir, 'data.nii.gz')
     
    if mask == '':
        mask_file = os.path.join(fs_data_dir, 'wb_mask.nii.gz')
    else:
        mask_file = os.path.join(fs_data_dir, '{}_mask.nii.gz'.format(mask))

    # load bold data and some header information so that we can save searchlight results later
    data_file = nib.load(data_file)
    bold_data = data_file.get_data()
    affine_mat = data_file.affine
    dimsize = data_file.header.get_zooms() 

    # load mask
    brain_mask = nib.load(mask_file)
    brain_mask = brain_mask.get_data()

    return bold_data, brain_mask, affine_mat, dimsize

def load_fs_label(sub_id, mask=''):
    # find file path
    sub = 'sub-%.2d' % (sub_id)
    input_dir = os.path.join(fs_data_dir, sub)
    label_file =  os.path.join(input_dir, 'label.npz')
    # load label
    label = np.load(label_file)
    label = label['label']
    return label
    
# Data Path
data_path = os.path.expanduser('~/searchlight_results')
# if not os.path.exists(data_path):
#     os.makedirs(data_path)

# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# load mask
mask_file = os.path.join(fs_data_dir, 'wb_mask.nii.gz')
mask = nib.load(mask_file)
mask = mask.get_data()

# Loop over subjects
data = []
bcvar = []
for sub_id in range(1,num_subj+1):
    if rank == 0:
        data_i, mask, affine_mat, dimsize = load_fs_data(sub_id)
        data.append(data_i)
    else:
        data.append(None)
    bcvar_i = load_fs_label(sub_id)
    bcvar.append(bcvar_i)

sl_rad = 1
max_blk_edge = 5
pool_size = 1

coords = np.where(mask)


# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)
# print("Setup searchlight inputs")
# print("Number of subjects: " + str(len(data)))
# print("Input data shape: " + str(data[0].shape))
# print("Input mask shape: " + str(mask.shape) + "\n")

# Distribute the information to the searchlights (preparing it to run)
sl.distribute(data, mask)

# Broadcast variables
sl.broadcast(bcvar)

# Set up the kernel function, in this case an SVM
def calc_svm(data, sl_mask, myrad, bcvar):
    accuracy = []
    sl_num_vx = sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2]
    num_epoch = data[0].shape[3]
    # Loop over subjects to leave each subject out once: 
    for idx in range(len(data)):
        # Pull out the data
        # Testing data
        data4D_test = data[idx]
        labels_test = bcvar[idx]
        bolddata_sl_test = data4D_test.reshape(sl_num_vx, num_epoch).T  
        
        # Training data
        labels_train = []
        bolddata_sl_train = np.empty((0, sl_num_vx))
        for train_id in range(len(data)):
            if train_id != idx:
                labels_train.extend(list(bcvar[train_id]))
                bolddata_sl_train = np.concatenate((bolddata_sl_train, data[train_id].reshape(sl_num_vx, num_epoch).T))
        labels_train = np.array(labels_train)
        
        # Train classifier
        clf = SVC(kernel='linear', C=1)
        clf.fit(bolddata_sl_train, labels_train)
        
        # Test classifier
        score = clf.score(bolddata_sl_test, labels_test)
        accuracy.append(score) 
        
    return accuracy

# Run the searchlight analysis
print("Begin SearchLight in rank %s\n" % rank)
all_sl_result = sl.run_searchlight(calc_svm, pool_size=pool_size)
print("End SearchLight in rank %s\n" % rank)

# Only save the data if this is the first core
if rank == 0: 
    all_sl_result = all_sl_result[mask==1]
    all_sl_result = [num_subj*[0] if not n else n for n in all_sl_result] # replace all None
    # The average result
    avg_vol = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))  
    
    # Loop over subjects
    for sub_id in range(1,num_subj+1):
        sl_result = [r[sub_id-1] for r in all_sl_result]
        # reshape
        result_vol = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))  
        result_vol[coords[0], coords[1], coords[2]] = sl_result   
        # Convert the output into what can be used
        result_vol = result_vol.astype('double')
        result_vol[np.isnan(result_vol)] = 0  # If there are nans we want this
        # Add the processed result_vol into avg_vol
        avg_vol += result_vol
        # Save the volume
        output_name = os.path.join(data_path, 'subj%s_whole_brain_SL.nii.gz' % (sub_id))
        sl_nii = nib.Nifti1Image(result_vol, affine_mat)
        hdr = sl_nii.header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
        nib.save(sl_nii, output_name)  # Save
    
    # Save the average result
    output_name = os.path.join(data_path, 'avg%s_whole_brain_SL.nii.gz' % (num_subj))
    sl_nii = nib.Nifti1Image(avg_vol/num_subj, affine_mat)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nib.save(sl_nii, output_name)  # Save    
    
    print('Finished searchlight')