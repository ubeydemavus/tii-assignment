from PIL import Image
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
from copy import deepcopy

# Increase the pixel size limit
Image.MAX_IMAGE_PIXELS = 1000000000  # Increase this value as needed


# Random Crop Generating Functions for testset "get_random_crop_coor_testset" and trainset "get_random_crop_coor_trainset"
# Calculations needed are different, thus two different functions. 
# But both function will output the same data structure for easier to management of data in the pipeline.
def get_random_crop_coor_trainset(rgb_base_array,thermal_base_array, rng = np.random.default_rng(), crop_size = 1536, patch_size = 512):
    """
        Given two base images and patch and crop sizes (rgb, thermal, crop_size, patch_size)
        generates random rgb_crop with crop_size, and a random thermal_patch with patch_size
    
        returns all coordinates that can be used to generate crops.
    """
    h, w, _ = rgb_base_array.shape
    
    rgb_random_top = rng.integers(0, h - crop_size)
    rgb_random_left = rng.integers(0, w - crop_size)
    
    patch_random_top = rng.integers(0, crop_size - patch_size)
    patch_random_left = rng.integers(0, crop_size - patch_size)
    
    # local center in rgb_crop (convention used x,y!)
    patch_rgb_crop_local_center = (patch_random_left + patch_size // 2 , patch_random_top + patch_size // 2)
    patch_rgb_base_center = rgb_random_left + patch_rgb_crop_local_center[0], rgb_random_top + patch_rgb_crop_local_center[1]
    
    
    crop_top = rgb_random_top
    crop_bottom = rgb_random_top + crop_size
    crop_left = rgb_random_left
    crop_right = rgb_random_left + crop_size
    
    patch_top = crop_top + patch_random_top 
    patch_bottom = crop_top + patch_random_top + patch_size
    patch_left = crop_left + patch_random_left
    patch_right = crop_left + patch_random_left + patch_size
    

    return {
        "patch_center_base":patch_rgb_base_center, 
        "patch_center_crop":patch_rgb_crop_local_center,
        "patch_base_top": patch_top,
        "patch_base_left": patch_left,
        "patch_base_bottom": patch_bottom, 
        "patch_base_right":patch_right,
        "rgb_base_top": crop_top,
        "rgb_base_left": crop_left,
        "rgb_base_bottom": crop_bottom,
        "rgb_base_right": crop_right,
        "patch_crop_top": patch_random_top,
        "patch_crop_left": patch_random_left,
        "crop_size": crop_size,
        "patch_size": patch_size
    }

def get_random_crop_coor_testset(center, 
                             rgb_base_array, 
                             thermal_patch,
                             rng = np.random.default_rng(),
                             thermal_patch_size = 512, 
                             crop_size = 1536):
    """
    Generates random crops from rgb base with their associated thermal patches in test dataset given the center (x, y)

    returns all coordinates that can be used to generate crops.
    """
    # Get height and width
    h, w, _ = rgb_base_array.shape
    # This is the center within the rgb_base (convention used in the dataset is x,y, not y,x!)
    center_x, center_y = center
    
    # Dimensions of the thermal patch
    thermal_top = center_y - thermal_patch_size // 2
    thermal_bottom = center_y + thermal_patch_size // 2
    thermal_left = center_x - thermal_patch_size // 2
    thermal_right = center_x + thermal_patch_size // 2

    # Calculate random crop bounds
    top_min = max(0, thermal_bottom - crop_size)
    top_max = min(h - crop_size, thermal_top)
    left_min = max(0, thermal_right - crop_size)
    left_max = min(w - crop_size, thermal_left)

    # Sample random top and left ensuring the thermal patch is included
    top = rng.integers(top_min, top_max + 1)
    left = rng.integers(left_min, left_max + 1)
    bottom = top + crop_size
    right = left + crop_size
    
    # Define the location of the thermal patch in the cropped RGB coordinates
    thermal_top_in_crop = thermal_top - top
    thermal_left_in_crop = thermal_left - left
    
        
    return {
        "patch_center_base":center, 
        "patch_center_crop": (thermal_left_in_crop + thermal_patch_size // 2 , thermal_top_in_crop + thermal_patch_size // 2),
        "patch_base_top": thermal_top,
        "patch_base_left": thermal_left,
        "patch_base_bottom": thermal_bottom,
        "patch_base_right": thermal_right, 
        "rgb_base_top": top,
        "rgb_base_left": left,
        "rgb_base_bottom": bottom,
        "rgb_base_right": right,
        "patch_crop_top": thermal_top_in_crop,
        "patch_crop_left": thermal_left_in_crop,
        "crop_size": crop_size,
        "patch_size": thermal_patch_size,
    }

def get_sample_images_trainset(rgb_base_array, thermal_base_array, item_dict, blend_ratio = 0.5):
    """
    given an sample item dict, returns rgb_crop, thermal_patch, overlay
    """
    # Get coordinates of rgb_crop from item dict
    crop_top = item_dict["rgb_base_top"]
    crop_bottom = item_dict["rgb_base_bottom"]
    crop_left = item_dict["rgb_base_left"]
    crop_right = item_dict["rgb_base_right"]

    # get coordinates of thermal_patch from item dict
    patch_top = item_dict["patch_base_top"]
    patch_bottom = item_dict["patch_base_bottom"]
    patch_left = item_dict["patch_base_left"]
    patch_right = item_dict["patch_base_right"]
    patch_size = item_dict["patch_size"]
    patch_crop_top = item_dict["patch_crop_top"]
    patch_crop_left = item_dict["patch_crop_left"]

    
    rgb_crop = rgb_base_array[crop_top:crop_bottom,crop_left:crop_right]
    thermal_patch = thermal_base_array[patch_top:patch_bottom, patch_left:patch_right] 
    
    # Overlay the thermal patch on the cropped RGB image
    overlay = rgb_crop.copy()
    
    # Ensure the thermal patch fits within the cropped RGB area
    overlay[
        patch_crop_top:patch_crop_top + patch_size ,
        patch_crop_left:patch_crop_left + patch_size
    ] = (1-blend_ratio) * overlay[
        patch_crop_top:patch_crop_top + patch_size ,
        patch_crop_left:patch_crop_left + patch_size
    ] + blend_ratio * thermal_patch  # Directly blend the RGB thermal patch

    return rgb_crop, thermal_patch, overlay
    
def get_sample_images_testset(rgb_base_array, thermal_patch, item_dict, blend_ratio = 0.5):
    """
    given an sample item dict, returns rgb_crop, thermal_patch, overlay
    """
    
    # Get coordinates of rgb_crop from item dict
    crop_top = item_dict["rgb_base_top"]
    crop_bottom = item_dict["rgb_base_bottom"]
    crop_left = item_dict["rgb_base_left"]
    crop_right = item_dict["rgb_base_right"]

    # get coordinates of thermal_patch from item dict
    thermal_top_in_crop = item_dict["patch_crop_top"]
    thermal_left_in_crop = item_dict["patch_crop_left"]
    thermal_patch_size = item_dict["patch_size"]

    # Extract the cropped RGB image
    rgb_crop = rgb_base_array[crop_top:crop_bottom,crop_left:crop_right]

    # Overlay the thermal patch on the cropped RGB image
    overlay = rgb_crop.copy()
    overlay[
        thermal_top_in_crop:thermal_top_in_crop + thermal_patch_size,
        thermal_left_in_crop:thermal_left_in_crop + thermal_patch_size,
        :
    ] = (1-blend_ratio) * overlay[
        thermal_top_in_crop:thermal_top_in_crop + thermal_patch_size,
        thermal_left_in_crop:thermal_left_in_crop + thermal_patch_size,
        :
    ] + blend_ratio * thermal_patch  # Directly blend the RGB thermal patch

    return rgb_crop, thermal_patch, overlay
    

# Pytorch dataset objects for trainset as well as testset. 
# Init functions differ, thus two different dataset class.
# However, their output will be the same.
class Trainset(Dataset):
    def __init__(self, 
                 rgb_base_path, 
                 thermal_base_path, 
                 num_samples = 1000, 
                 rgb_transforms = None, 
                 thermal_tranforms=None, 
                 rng = np.random.default_rng(),
                 patch_size=512, 
                 crop_size=1536):
        # set local random generator if provided.
        self.rng = rng
        # Get image Convert RGB PIL image to numpy array
        self.rgb_base_array = np.array(Image.open(rgb_base_path).convert("RGB"))
        self.thermal_base_array = np.array(Image.open(thermal_base_path).convert("RGB")) 
        # Get sizes
        self.crop_size = crop_size
        self.patch_size = patch_size
        # Get transformations (For homography transformations)
        self.rgb_transforms = rgb_transforms
        self.thermal_tranforms = thermal_tranforms
        # Get how many samples to generate
        self.num_samples = num_samples
        # Generated sampled will be hold in a list
        self.samples = []
        # Generate samples. 
        self._getsamples()
        
    def __len__(self,):
        return self.num_samples

    def _getsamples(self):
        # Iteratively generate random crop coordinates from the base images, and store them in samples.
        for item in range(self.num_samples):
            self.samples.append(get_random_crop_coor_trainset(self.rgb_base_array, 
                                                         self.thermal_base_array, 
                                                         self.rng,
                                                         self.crop_size, 
                                                         self.patch_size)
                               )    

    def __getitem__(self,idx):
        if isinstance(idx, slice):
            # Process the slice and return a list of items
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.samples)))]
        elif isinstance(idx, int):
            # Process single index as usual
            item = self.samples[idx]
    
            # lazily creating images here to avoid high memory overhead
            rgb_crop, thermal_patch, overlay = get_sample_images_trainset(self.rgb_base_array, self.thermal_base_array, item)
    
            # Deep copy item to avoid increasing dataset object memory usage
            item_to_return = deepcopy(item)
            
            item_to_return["rgb_crop"] = rgb_crop
            item_to_return["thermal_patch"] = thermal_patch
            item_to_return["overlay"] = overlay
            
            # Apply rgb image transformations if applies (maybe contrast enhancement etc..)
            if self.rgb_transforms:
                item_to_return = self.rgb_transforms(item_to_return)
                
            # Apply thermal patch transformations if applied (Homography transformations)
            if self.thermal_tranforms:
                # Necessary coordinate transformations will be inside thermal_transforms pipeline.
                item_to_return = self.thermal_tranforms(item_to_return)
    
            return item_to_return


class Testset(Dataset):
    def __init__(self, 
                 rgb_base_path, 
                 thermal_patches, 
                 centers,
                 num_samples = 1000,
                 rgb_transforms = None, 
                 thermal_tranforms=None, 
                 rng = np.random.default_rng(),
                 patch_size=512, 
                 crop_size=1536):
        # set local random generator
        self.rng = rng
        # Get Image and Convert RGB PIL image to numpy array
        self.rgb_base_array = np.array(Image.open(rgb_base_path).convert("RGB"))
        # get thermal patches
        self.thermal_patches = thermal_patches
        # get number of samples to generate
        self.num_samples = num_samples
        # get centers
        self.centers = centers 
        # Get sizes
        self.crop_size = crop_size
        self.patch_size = patch_size
        # Get transformations (For homography transformations)
        self.rgb_transforms = rgb_transforms
        self.thermal_tranforms = thermal_tranforms
        # Generated sampled will be hold in a list
        self.samples = []
        # Generate samples. 
        self._getsamples()
        
    def __len__(self,):
        return self.num_samples

    def _getsamples(self):
        # Iteratively generate random crops from the base images.
        for item in range(self.num_samples):
            # Randomly select a center to generate thermal patch with a random rgb crop.
            idx = self.rng.integers(0,len(self.centers))
            item = get_random_crop_coor_testset(self.centers[idx],
                                                        self.rgb_base_array, 
                                                        self.thermal_patches[idx], 
                                                        self.rng,
                                                        thermal_patch_size = self.patch_size, 
                                                        crop_size = self.crop_size)
            item["item_id"] = idx
            self.samples.append(item)    
            
    def __getitem__(self,idx):
        if isinstance(idx, slice):
            # Process the slice and return a list of items
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.samples)))]
        elif isinstance(idx, int):
            # Process single index as usual
            item = self.samples[idx]
    
            # lazily creating images here to avoid high memory overhead
            rgb_crop, thermal_patch, overlay = get_sample_images_testset(self.rgb_base_array, self.thermal_patches[item["item_id"]], item)
    
            # Deep copy item to avoid increasing dataset object memory requirements. 
            item_to_return = deepcopy(item)
            
            item_to_return["rgb_crop"] = rgb_crop
            item_to_return["thermal_patch"] = thermal_patch
            item_to_return["overlay"] = overlay
            
            del item_to_return["item_id"] # not needed at output.
            
            # Apply rgb image transformations if applies (maybe contrast enhancement etc..)
            if self.rgb_transforms:
                item_to_return = self.rgb_transforms(item_to_return)
            # Apply thermal patch transformations if applied (Homography transformations)
            if self.thermal_tranforms:
                # Necessary coordinate transformations will be inside thermal_transforms pipeline.
                item_to_return = self.thermal_tranforms(item_to_return)
    
            return item_to_return

if __name__ == '__main__':
    # if running as a script, call functions and dataset & dataloader objects for testing purposes.
    # shows figues in a blocking manner, one windows at a time.
    import matplotlib.pyplot as plt

    print("Loading uav.pkl")
    # Load thermal patches and centers from the pickle file
    with open("datasets/uav.pkl", "rb") as f:
        uav_data = joblib.load(f)
    thermal_patches = uav_data['anchor']  
    centers = uav_data['center']

    print("Loading png images")
    # Load RGB and thermal images
    rgb_zone1 = Image.open("datasets/rgb_zone1.png").convert("RGB")
    thermal_zone1 = Image.open("datasets/thermal_zone1.png").convert("RGB")
    rgb_zone2 = Image.open("datasets/rgb_zone2.png").convert("RGB")
    
    # Convert images to numpy arrays
    rgb_zone1_array = np.array(rgb_zone1)
    thermal_zone1_array = np.array(thermal_zone1)
    rgb_zone2_array = np.array(rgb_zone2)

    #set random seeds for cropping as well as shuffling
    torch.manual_seed(0) # for dataloader reproducibility 
    rng = np.random.default_rng(seed = 0) # for cropping reproducibility

    print("Generating trainset")
    sample_train_dataset = Trainset(
        rgb_base_path = "datasets/rgb_zone1.png",
        thermal_base_path = "datasets/thermal_zone1.png",
        num_samples = 20,
        rng = rng
    )
    
    sample_train_dataloader = DataLoader(sample_train_dataset,batch_size=10, drop_last=False, shuffle=True)
    items = next(iter(sample_train_dataloader))
    plt.figure()
    plt.imshow(items["overlay"][6])
    plt.title("Trainset object test")
    plt.show()

    #set random seeds for cropping as well as shuffling
    torch.manual_seed(0) 
    rng = np.random.default_rng(seed = 0)

    print("Generating testset")
    sample_test_dataset = Testset(
        rgb_base_path = "datasets/rgb_zone2.png",
        thermal_patches= thermal_patches,
        centers=centers,
        num_samples = 20,
        rng = rng
    )
    
    sample_test_dataloader = DataLoader(sample_test_dataset,batch_size=10, drop_last=False, shuffle=True)
    items = next(iter(sample_test_dataloader))
    plt.figure()
    plt.imshow(items["overlay"][0])
    plt.title("Testset object test")
    plt.show()

    