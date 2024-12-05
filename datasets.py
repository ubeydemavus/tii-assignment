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


# Create keypoint dataset.
class KeypointsDataset(Dataset):
    def __init__(self, samples, max_num_matches = 10, crop_size = 1536, patch_size = 512):
        self.samples = samples
        self.max_num_matches = max_num_matches
        self.crop_size = crop_size
        self.patch_size = patch_size

    def __len__(self,):
        return len(self.samples)

    def __getitem__(self,idx):
        sample = self.samples[idx]
        
        # if no matches found, create all zero input keypoints and attention mask of zeros (basically we are ignoring the sample)
        if sample["points_rgb_crop"].size == 0:
            keypoints_rgb = np.zeros((self.max_num_matches,2))
            keypoints_patch = np.zeros((self.max_num_matches,2))
            mask = np.ones(self.max_num_matches) # we will ignore all of them (could cause instability during training. Fix later)
            center_point = (-1,-1)
        # otherwise select self.max_num_matches of matches 
        else:
            # order keypoints from highest matching score to lowest
            sorted_indexes = np.argsort(sample["points_matching_scores"])[::-1]
    
            # normalize points
            keypoints_rgb = sample["points_rgb_crop"][sorted_indexes] / self.crop_size 
            keypoints_patch = sample["points_thermal_patch"][sorted_indexes] / self.patch_size 
            center_point =  np.asarray(sample["patch_center_crop"]) / self.crop_size
    
            # truncate if more matches exists, pad 0's otherwise # so that we can batch train.
            if keypoints_rgb.shape[0] >= self.max_num_matches:
                keypoints_rgb = keypoints_rgb[:self.max_num_matches,:]
                keypoints_patch = keypoints_patch[:self.max_num_matches,:]
                mask = np.zeros(self.max_num_matches) # No key_padding_mask 
            else:
                pad_length = self.max_num_matches - keypoints_rgb.shape[0]
                keypoints_rgb = np.pad(keypoints_rgb,((0,pad_length),(0,0)), constant_values = 0)
                keypoints_patch = np.pad(keypoints_patch,((0,pad_length),(0,0)), constant_values = 0)
                mask = np.zeros(self.max_num_matches)
                mask[-pad_length:] = 1. # mask padded values to zero, so we can ignore in cross attention.
                
        return (keypoints_rgb, keypoints_patch, mask, center_point)

# Collate function to handle batch processing (masks caused some issues without custom collate_fn)
def collate_fn(batch):
    # Unzip the batch into separate lists
    keypoints_a, keypoints_b, masks, centers = zip(*batch)

    # Convert to tensors first
    keypoints_a = torch.tensor(keypoints_a, dtype=torch.float32)
    keypoints_b = torch.tensor(keypoints_b, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.bool)
    centers = torch.tensor(centers, dtype=torch.float32)

    # Identify fully masked samples (all True in the mask)
    valid_samples = [i for i, mask in enumerate(masks) if not mask.all()]

    # If there are no valid samples, return empty tensors
    if len(valid_samples) == 0:
        return (
            torch.empty(0, keypoints_a.size(1), keypoints_a.size(2)),
            torch.empty(0, keypoints_b.size(1), keypoints_b.size(2)),
            torch.empty(0, masks.size(1)),
            torch.empty(0, centers.size(1))
        )

    # Filter out the invalid samples
    keypoints_a = keypoints_a[valid_samples]
    keypoints_b = keypoints_b[valid_samples]
    masks = masks[valid_samples]
    centers = centers[valid_samples]

    return keypoints_a, keypoints_b, masks, centers

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

    print("Testing KeypointsDataset")
    #pseudo samples list
    samples = [{'patch_center_base': (7514, 6999),
                  'patch_center_crop': (844, 518),
                  'patch_base_top': 6743,
                  'patch_base_left': 7258,
                  'patch_base_bottom': 7255,
                  'patch_base_right': 7770,
                  'rgb_base_top': 6481,
                  'rgb_base_left': 6670,
                  'rgb_base_bottom': 8017,
                  'rgb_base_right': 8206,
                  'patch_crop_top': 262,
                  'patch_crop_left': 588,
                  'crop_size': 1536,
                  'patch_size': 512,
                  'points_rgb_crop': np.array([[ 808.92035,  593.7069 ],
                         [ 736.7803 ,  579.31006],
                         [1087.4019 ,  619.316  ],
                         [ 889.5041 ,  611.5062 ],
                         [ 921.2839 ,  448.93927],
                         [ 656.05725,  718.3458 ]],),
                  'points_thermal_patch': np.asarray([[220.81586, 340.06625],
                         [149.80637, 326.26135],
                         [502.6948 , 367.3821 ],
                         [300.45184, 359.75287],
                         [331.7768 , 194.84872],
                         [ 67.13458, 468.08237]],),
              'points_matching_scores': np.asarray([0.9142908 , 0.68882734, 0.9817511 , 0.5173324 , 0.91734767,
                     0.99766195 ],)},
                 {'patch_center_base': (5890, 4207),
                  'patch_center_crop': (262, 1067),
                  'patch_base_top': 3951,
                  'patch_base_left': 5634,
                  'patch_base_bottom': 4463,
                  'patch_base_right': 6146,
                  'rgb_base_top': 3140,
                  'rgb_base_left': 5628,
                  'rgb_base_bottom': 4676,
                  'rgb_base_right': 7164,
                  'patch_crop_top': 811,
                  'patch_crop_left': 6,
                  'crop_size': 1536,
                  'patch_size': 512,
                  'points_rgb_crop': np.asarray([[518.2568  , 187.98552 ],
                         [525.58984 , 196.91994 ],
                         [395.95538 , 374.6199  ],
                         [400.37546 , 383.19412 ],
                         [435.04614 , 383.39398 ],
                         [117.75293 , 447.2601  ],
                         [ 87.29519 , 453.73352 ],
                         [ 21.557617, 477.45135 ],
                         [231.234   , 514.7124  ],
                         [ 73.79296 , 552.41766 ],
                         [276.24652 , 588.8331  ],
                         [339.25916 , 597.2624  ],
                         [349.48584 , 607.76996 ],
                         [338.14423 , 609.52045 ],
                         [334.7986  , 621.5207  ],
                         [351.56705 , 622.5673  ],
                         [271.7867  , 625.65393 ],
                         [274.2113  , 644.06616 ],
                         [ 76.24239 , 650.9151  ],
                         [196.55463 , 661.7084  ],
                         [374.10413 , 661.5995  ],
                         [ 79.56704 , 664.937   ],
                         [207.5444  , 673.50555 ],
                         [163.84459 , 676.737   ],
                         [147.22214 , 689.2918  ],
                         [147.24039 , 705.5265  ],
                         [180.27274 , 717.1914  ],
                         [114.13895 , 737.7818  ],
                         [163.71434 , 745.3049  ],
                         [113.01138 , 748.89215 ],
                         [ 88.69527 , 749.9638  ],
                         [136.584   , 750.2643  ],
                         [109.273926, 763.4987  ],
                         [204.1362  , 776.17615 ],
                         [117.376465, 787.9152  ]],),
                  'points_thermal_patch': np.asarray([[309.2712  ,  14.379437],
                         [316.23126 ,  25.800707],
                         [322.32187 , 334.45532 ],
                         [275.217   , 339.71915 ],
                         [310.12003 , 354.48462 ],
                         [508.74884 , 173.22786 ],
                         [247.27551 , 139.69243 ],
                         [128.27837 , 366.9138  ],
                         [307.18063 , 378.64728 ],
                         [426.17065 , 370.67392 ],
                         [500.1454  , 381.1269  ],
                         [476.50174 , 388.94174 ],
                         [482.69617 , 392.78992 ],
                         [477.8264  , 396.0303  ],
                         [474.41412 , 403.70468 ],
                         [482.41754 , 409.06894 ],
                         [462.5746  , 394.46024 ],
                         [466.2682  , 405.65103 ],
                         [256.6701  , 441.28485 ],
                         [417.30417 , 424.52515 ],
                         [503.2133  , 440.2345  ],
                         [150.24254 , 459.63507 ],
                         [427.16068 , 428.50342 ],
                         [418.75668 , 438.5565  ],
                         [364.24734 , 443.9187  ],
                         [377.7618  , 455.78876 ],
                         [363.49713 , 470.48935 ],
                         [300.1968  , 475.1791  ],
                         [401.69357 , 494.29874 ],
                         [239.34065 , 504.6395  ],
                         [227.21738 , 505.69983 ],
                         [383.23178 , 490.2425  ],
                         [278.083   , 479.75757 ],
                         [416.83096 , 480.36148 ],
                         [352.07822 , 498.57797 ]],),
                  'points_matching_scores': np.asarray([0.13859066, 0.20301743, 0.14682397, 0.45579177, 0.14860633,
                         0.16324554, 0.21792185, 0.10593864, 0.10732379, 0.34075406,
                         0.12781201, 0.6104509 , 0.10385362, 0.41758057, 0.2291022 ,
                         0.124134  , 0.10231926, 0.33154267, 0.15009704, 0.53715646,
                         0.15291016, 0.22681785, 0.23690207, 0.10788915, 0.17242393,
                         0.14125662, 0.12352585, 0.104253  , 0.16773959, 0.20628694,
                         0.10257275, 0.18559173, 0.37215924, 0.13309997, 0.14071223],)}]
    
    for rgb_keypoints,patch_keypoints,mask,center in KeypointsDataset(samples,max_num_matches=10):
        print(f"""
        {rgb_keypoints=}
        {patch_keypoints=}
        {mask=}
        {center=}
        ----
        """)
    