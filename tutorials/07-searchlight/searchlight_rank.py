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

num_subj = 1

# Load and perpare data for one subject
def load_fs_data(sub_id, mask=''):
    # find file path
    sub = 'sub-%.2d' % (sub_id)
    input_dir = os.path.join(fs_data_dir, sub)
    data_file = os.path.join(input_dir, 'data.nii.gz')
    label_file =  os.path.join(input_dir, 'label.npz') 
    if mask == '':
        mask_file = os.path.join(fs_data_dir, 'wb_mask.nii.gz')
    else:
        mask_file = os.path.join(fs_data_dir, '{}_mask.nii.gz'.format(mask))

    # load bold data and some header information so that we can save searchlight results later
    data_file = nib.load(data_file)
    bold_data = data_file.get_data()
    affine_mat = data_file.affine
    dimsize = data_file.header.get_zooms() 
    
    # load label
    label = np.load(label_file)
    label = label['label']

    # load mask
    brain_mask = nib.load(mask_file)
    brain_mask = brain_mask.get_data()

    return bold_data, label, brain_mask, affine_mat, dimsize

# Data Path
data_path = os.path.expanduser('~/searchlight_results')
# if not os.path.exists(data_path):
#     os.makedirs(data_path)

# Loop over subjects
data = []
bcvar = []
for sub_id in range(1,num_subj+1):
    data_i, bcvar_i, mask, affine_mat, dimsize = load_fs_data(sub_id)
    data.append(data_i)
    bcvar.append(bcvar_i)

sl_rad = 1
max_blk_edge = 5
pool_size = 1

coords = np.where(mask)

# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)
print("Setup searchlight inputs")
print("Number of subjects: " + str(len(data)))
print("Input data shape: " + str(data[0].shape))
print("Input mask shape: " + str(mask.shape) + "\n")

# Distribute the information to the searchlights (preparing it to run)
sl.distribute(data, mask)

# Broadcast variables
sl.broadcast(bcvar)

# Set up the kernel function, in this case an SVM
def calc_rank(data, sl_mask, myrad, bcvar):
    # Pull out the MPI information
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size        
    return [rank]

# Run the searchlight analysis
print("Begin SearchLight in rank %s\n" % rank)
all_sl_result = sl.run_searchlight(calc_rank, pool_size=pool_size)
print("End SearchLight in rank %s\n" % rank)

# Only save the data if this is the first core
if rank == 0: 
    all_sl_result = all_sl_result[mask==1]
    all_sl_result = [num_subj*[0] if not n else n for n in all_sl_result] # replace all None
    
    # Loop over subjects
    for sub_id in range(1,num_subj+1):
        sl_result = [r[sub_id-1] for r in all_sl_result]
        # reshape
        result_vol = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))  
        result_vol[coords[0], coords[1], coords[2]] = sl_result   
        # Convert the output into what can be used
        result_vol = result_vol.astype('double')
        result_vol[np.isnan(result_vol)] = 0  # If there are nans we want this
        # Save the volume
        output_name = os.path.join(data_path, 'rank_whole_brain_SL.nii.gz' )
        sl_nii = nib.Nifti1Image(result_vol, affine_mat)
        hdr = sl_nii.header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
        nib.save(sl_nii, output_name)  # Save
      
    print('Finished searchlight')