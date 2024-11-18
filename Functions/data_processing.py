# Required packages
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



def get_patient_ids(path): 
    """
    Obtain all the patients' ids that are inside a folder.
    """   
    # List all items in the directory, filtering for directories only
    folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    # Remove everything after the underscore in each folder name
    cleaned_names = [name.split('_')[0] for name in folder_names]
    
    return cleaned_names



def extract_data(source_folder, patients, split=None, annotated=False, data=None, train=False):    
    """
    The following function creates a dictionary where:

    - Keys: Represent individual patients.
    - Values: For each patient, there is another dictionary:
        - Patient Label: Derived from `PatientDiagnosis.csv`.
        - Patch List: Contains all associated patches for that patient, where each patch is a dictionary with:
            - Patch Label: Taken from annotated patches in the `.xlsx` file.
            - Path to Image: The file path to the patch image.

    The helper function `load_image(patch)` is provided to load images from these specified paths.

    ---

    The function is designed to work with the `Annotated`, `Cropped`, and `HoldOut` splits.
    
    - Parameters:
        - `split=None`: It should be passed as a list of patient ids that compose a data split. The function will only
                        retrieve the patients within that list. Otherwise, the function scraps all the patients of a folder.
        - `annotated=False`: Set to `True` when processing the `Annotated` split to handle patch labels. 
                             The `data` parameter should contain the DataFrame with labelled patch information.
        - `train=False`: Set to `True` when creating a training set for the autoencoder. 
                         In this mode, the function will only include information from healthy patients.
    """

    # Dictionary to store the results
    patient_data = {}
    num_patients = 0

    # Step 1: Loop through each unique patient code
    for patient in patients['CODI'].unique():

        # Step 2: Check if data has been splitted and avoid patients outside the split
        if split is not None:
            if patient not in split:
                continue

        # Step 3: Construct possible paths for the patient folder and check existence
        pat_path = os.path.join(source_folder, f"{patient}_0")
        if not os.path.exists(pat_path):
            pat_path = os.path.join(source_folder, f"{patient}_1")
            if not os.path.exists(pat_path):
                continue

        # Step 4: List all image files with accepted extensions in the patient folder
        image_files = [f for f in os.listdir(pat_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue  # Skip patients with no images

        # Step 5: Determine the patient-level label
        densitat_value = patients.loc[patients['CODI'] == patient, 'DENSITAT'].values[0]
        patient_label = 1 if densitat_value != 'NEGATIVA' else -1
        if train and patient_label == 1: # Skip infected patients if train = True
            continue

        # Step 6: Process each patch in the patient's folder
        patient_entries = []

        for filename in image_files:
            src_path = os.path.join(pat_path, filename)

            if annotated: # Patches have labels
                window_id = filename.split('_')[0].split('.')[0].lstrip('0') or '0'

                # Extract information by matching patient and window ID
                label_info = data[(data['Pat_ID'] == patient) & (data['Window_ID'] == int(window_id))]
                if label_info.empty:
                    continue  # Skip if no label is found

                # Get the presence label (1 for bacteria, -1 for no bacteria)
                label = label_info.iloc[0]['Presence']
                entry = {'img_path': src_path, 'label': label}

            else:
                # No labels on patches
                entry = {'img_path': src_path, 'label': None}

            # Append the path entry to this patch's data
            patient_entries.append(entry)

        # Step 7: Add the patient label and patches information to the patient data
        patient_data[patient] = {
            'images': patient_entries,
            'patient_label': patient_label
        }
        num_patients += 1

    return patient_data

# Helper function to load image on demand
def load_image(entry):
    ''' Lazy-loads an image given an entry with an 'img_path' '''
    return Image.open(entry['img_path']).convert('RGB')



class PILImageDataset(Dataset):
    """
    This class processes patches from patient data, applying transformations to each image as it is accessed. 
    It pulls individual patches from the provided patient dataset, organizing them into a list format. 
    Each patch is stored with its associated `patient_id` and `index`, enabling posterior identification.
    """
    def __init__(self, patient_data, transform):
        self.patches = self.obtain_patches(patient_data)
        self.transform = transform

    def obtain_patches(self, patient_data):
        patches = []

        # Iterate over each patient and their data
        for patient_id, entries in patient_data.items():
            image_entries = entries['images']

            # Add a tuple containing patient_id, index, and patch
            for index, patch in enumerate(image_entries):
                patches.append((patient_id, index, patch))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patient_id, index, patch = self.patches[idx]
        patch = self.transform(load_image(patch))
        return patch, patient_id, index  # Return the image along with its patient_id and index
    


def create_X_y(patient_data, patch_level=False):
    """    
    Extracts features and labels for model training and inference with sklearn
    If `patch_level` is set to `True`, the function retrieves the features and labels of the patches.
    Otherwise, it retrieves the features and labels of the patients.
    """
    X = []  # Feature list
    y = []  # Label list
    metadata = []  # To track which patient and patch each feature corresponds to
    
    # Collect features, labels, and indices for all patches
    for patient_id, entry in patient_data.items():
        if patch_level:
            for idx, patch in enumerate(entry['images']):            
                X.append(patch['features'])
                y.append(patch['label'])
                metadata.append((patient_id, idx))
        
        else:
            X.append(entry['patient_features'])
            y.append(entry['patient_label'])
            metadata.append(patient_id)

    return np.array(X), np.array(y), metadata



def add_patient_features(patient_data):
    """
    Based on the patch features, this function creates four features for each patient:
        - Total Red Pixel Count: The aggregated number of red pixels in all the patient's patches.
        - Total Pixel Difference: The aggregated absolute difference in the number of red.
        - Mean Percentage Difference: The aggregated percentage difference in red pixels.
        - Percentage of positive patches: The relative amount of positive patches that the patient has.
    """
    for _, entry in patient_data.items():
        # Initialize values for aggregation
        total_red_pixels = 0
        total_difference = 0
        total_percentage_diff = 0
        num_positive_patches = 0
        num_patches = 0

        # Iterate through each patch (dictionary) for the patient
        for patch in entry['images']:            
            num_patches += 1
            # Accumulate red pixel counts and differences
            total_red_pixels += patch['features'][0]
            total_difference += patch['features'][1]
            total_percentage_diff += patch['features'][2]
            
            # Count positive patches
            if patch['prediction'] == 1:
                num_positive_patches += 1
        
        # Calculate mean percentage difference
        mean_percentage_diff = total_percentage_diff / num_patches if num_patches > 0 else 0
        
        # Calculate the percentage of positive patches
        positive_patch_percentage = (num_positive_patches / num_patches) if num_patches > 0 else 0
        
        # Create the patient_features list with the aggregated values and the new feature
        patient_features = [
            total_red_pixels,            # Aggregated red pixels
            total_difference,            # Aggregated difference
            mean_percentage_diff,        # Mean percentage difference
            positive_patch_percentage    # Percentage of positive patches (based on patch classification)
        ]
        
        # Append patient_features to the list for each patient
        entry['patient_features'] = patient_features   