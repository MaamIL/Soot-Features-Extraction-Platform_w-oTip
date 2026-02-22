import numpy as np
import os
import PIL.Image as Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from Logger import CustomLogger


# Custom Dataset: each subfolder in root_dir should contain CFDImage.mat and sootCalculation.mat
class FlameDataset(Dataset):
    """
    Custom dataset for loading flame simulation data.
    Each sample is expected to be in a subdirectory containing:
    - CFDImage.mat: Contains the CFD image data.
    - sootCalculation.mat: Contains the soot and temperature data.
    """
    def __init__(self, config):
        """
        Initialize the dataset with the given configuration.
        Args:
            config (object): Configuration object containing dataset parameters.
        """
        self.config = config
        self.logger = CustomLogger(config.log_filename, self.__class__.__name__).get_logger()
        self.logger.info("FlameDataset initialized with the provided configuration")
        self.sample_dirs = [os.path.join(config.root_dir, d) for d in os.listdir(config.root_dir)
                            if os.path.isdir(os.path.join(config.root_dir, d))] # List all subdirectories (each is one sample)
        self.logger.info(f"Found {len(self.sample_dirs)} samples in the dataset.")
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.sample_dirs)
    
    # def compute_mean_std(self):
    #     """
    #       Compute mean and standard deviation of the dataset images.
    #     Returns:
    #       tuple: Mean and standard deviation of the dataset images.
    #      """
    #     mean = 0.0
    #     std = 0.0
    #     for idx in range(len(self)):
    #         image, _ = self[idx]  # Get the image from __getitem__
    #         mean += image.mean([1, 2])  # Compute mean across height and width for each channel
    #         std += image.std([1, 2])    # Compute std across height and width for each channel

    #     mean /= len(self)
    #     std /= len(self)
    #     self.logger.info(f"Computed Mean: {mean}, Std: {std}")
    #     return mean, std
    
    def _getFv_(self, soot_mat):
        """
        This function extracts and processes the fv array from the sootCalculation mat.
        initialize small values to 0.0 and normalize if required. Pad to the desired shape.        
        Args:
            soot_mat (dict): Loaded .mat file containing soot data.
        Returns:
            np.ndarray: Normalized and padded fv array.
        """
        fv = soot_mat["fv"]#fv = soot_mat["fvPixelated"]
        # Set values smaller than 0.1 to 0
        fv[fv < self.config.setFvValZero] = 0.0
        #Normalize fv
        if(self.config.isNorm):
            fv = (fv - self.config.global_fv_min)/max((self.config.global_fv_max-self.config.global_fv_min), 1e-6)
        # Pad fv to the desired shape (202, 92)
        fv = self.pad_or_crop_to_shape_output(fv, self.config.output_shape, "Fv")  # Pad or crop to the desired shape
        # fv = np.pad(fv, ((0, self.config.output_shape[0] - fv.shape[0]), (0, self.config.output_shape[1] - fv.shape[1])), mode='constant', constant_values=0)    
        return fv
    
    def _getT_(self, soot_mat):
        """
        This function extracts and processes the T array from the sootCalculation mat.
        initialize small values to 300.0 and normalize if required. Pad to the desired shape.        
        Args:
            soot_mat (dict): Loaded .mat file containing soot data.
        Returns:
            np.ndarray: Normalized and padded T array.
        """
        T = soot_mat["T"]  #T = soot_mat["TPixelated"]
        T[T < self.config.setTValZero] = 300.0    
        #Normalize T
        if(self.config.isNorm):
            T = (T - self.config.global_T_min)/max((self.config.global_T_max-self.config.global_T_min), 1e-6)
            
        #Temperature padding needs to be with values of 300 (kalvin as the minimum value is 300)- put 0.0 because it is normalized
        T = self.pad_or_crop_to_shape_output(T, self.config.output_shape, "T")  # Pad or crop to the desired shape
        # T  = np.pad(T,((0, self.config.output_shape[0] - T.shape[0]), (0, self.config.output_shape[1] - T.shape[1])), mode='constant', constant_values=0.0)             
        return T
    
    def pad_or_crop_to_shape_output(self, arr, target_shape, tORfv):
        """
        Pads or crops the arr to match the target shape.
        Only zero-valued margins are cropped if the image is too large.
        
        Args:
            arr (np.ndarray): Shape (H, W, C)
            target_shape (tuple): (C, target_H, target_W)

        Returns:
            np.ndarray: Padded or cropped image of shape (target_H, target_W, C)
        """
        target_H, target_W = target_shape[0], target_shape[1]
        H, W = arr.shape
        orig_H, orig_W = H, W
        # --- CROP FROM MARGINS IF TOO BIG ---
        if H > target_H:
            # Crop zero rows from top and bottom
            while H > target_H and np.all(arr[0, :] == 0):
                arr = arr[1:, :]
                H -= 1
            while H > target_H and np.all(arr[-1, :] == 0):
                arr = arr[:-1, :]
                H -= 1
            # Now crop from the end (bottom) if still too large
            while H > target_H:
                arr = arr[:-1, :]
                H -= 1

        if W > target_W:
            # Crop zero columns from left and right
            while W > target_W and np.all(arr[:, 0] == 0):
                arr = arr[:, 1:]
                W -= 1
            while W > target_W and np.all(arr[:, -1] == 0):
                arr = arr[:, :-1]
                W -= 1
            # Now crop from the end (right) if still too large
            while W > target_W:
                arr = arr[:, :-1]
                W -= 1

        # --- PAD IF TOO SMALL ---
        pad_H = max(0, target_H - arr.shape[0])
        pad_W = max(0, target_W - arr.shape[1])
        
        arr = np.pad(
            arr,
            ((0, pad_H), (0, pad_W)),
            mode='constant',
            constant_values=0
        )
        if self.config.MODE != "Train" and self.config.MODE != "TrainSaveAllData":
            self.logger.info(f"Padded/Cropped {tORfv} to shape: {arr.shape} from original shape: {(orig_H, orig_W)} to target shape: {target_shape}")
            self.logger.info(f"minimum value in {tORfv} array: {np.min(arr[arr > 0])}, maximum value: {np.max(arr)}")
        return arr

    def pad_or_crop_to_shape_img(self, image_array, target_shape):
        """
        Pads or crops the image_array to match the target shape.
        Only zero-valued margins are cropped if the image is too large.
        
        Args:
            image_array (np.ndarray): Shape (H, W, C)
            target_shape (tuple): (C, target_H, target_W)

        Returns:
            np.ndarray: Padded or cropped image of shape (target_H, target_W, C)
        """
        target_H, target_W = target_shape[1], target_shape[2]
        H, W, C = image_array.shape
        orig_H, orig_W, orig_C = H, W, C
        # --- CROP FROM MARGINS IF TOO BIG ---
        if H > target_H:
            # Crop zero rows from top and bottom- comment this in order to get the "lower" flame (without taking out zeros from the base)
            while H > target_H and np.all(image_array[0, :, :] == 0):
                image_array = image_array[1:, :, :]
                H -= 1
            
            while H > target_H and np.all(image_array[-1, :, :] == 0):
                image_array = image_array[:-1, :, :]
                H -= 1
            # Now crop from the end (bottom) if still too large
            while H > target_H:
                image_array = image_array[:-1, :, :]
                H -= 1

        if W > target_W:
            # Crop zero columns from left and right
            while W > target_W and np.all(image_array[:, 0, :] == 0):
                image_array = image_array[:, 1:, :]
                W -= 1
            while W > target_W and np.all(image_array[:, -1, :] == 0):
                image_array = image_array[:, :-1, :]
                W -= 1

        # --- PAD IF TOO SMALL ---
        pad_H = max(0, target_H - image_array.shape[0])
        pad_W = max(0, target_W - image_array.shape[1])
        
        # #pad on top (pad_H, 0) and right (0, pad_W)
        # image_array = np.pad(
        #     image_array,
        #     ((pad_H, 0), (0, pad_W), (0, 0)),
        #     mode='constant',
        #     constant_values=0
        # )
        # #pad on bottom (0, pad_H) and right (0, pad_W)
        image_array = np.pad(
            image_array,
            ((0, pad_H), (0, pad_W), (0, 0)),
            mode='constant',
            constant_values=0
        )
        if self.config.MODE != "Train" and self.config.MODE != "TrainSaveAllData":
            self.logger.info(f"Padded/Cropped image to shape: {image_array.shape} from original shape: {(orig_H, orig_W, orig_C)} to target shape: {target_shape}")
            self.logger.info(f"minimum value in image array: {np.min(image_array[image_array > 0])}, maximum value: {np.max(image_array)}")
            
        return image_array

    def if_needsFlipud(self, image_array, num_rows=3):
        """
        Check if the image array should be flipped vertically.
        We want the wider base to be at the bottom of the image.
        This function checks the first and last N non-zero rows to determine which base is wider.
        Args:
            image_array (np.ndarray): The image array to check.
        Returns:
            bool: True if the image should be flipped, False otherwise.
        """        
        # Convert RGB to grayscale for non-zero checking
        grayscale = np.any(image_array != 0, axis=2)  # shape: (H, W)
                
        # Find indices of rows that are not all zero
        nonzero_row_indices = np.where(np.any(grayscale, axis=1))[0]
        
        if len(nonzero_row_indices) < num_rows:
            raise ValueError("Not enough non-zero rows.")

        # First N non-zero rows
        first_rows = grayscale[nonzero_row_indices[:num_rows]]        
        first_widths = [np.count_nonzero(row) for row in first_rows]
        first_max_width = max(first_widths)

        # Last N non-zero rows
        last_rows = grayscale[nonzero_row_indices[-num_rows:]]
        last_widths = [np.count_nonzero(row) for row in last_rows]
        last_max_width = max(last_widths)

        return last_max_width >= first_max_width # we want image upside down (top of flame (image) is wider)
        # return first_max_width > last_max_width  # we want image right side up (bottom of flame (image) is wider)  
        
    def _getImage_(self, sample_dir):
        """
        Load and preprocess the CFD image from the given path.
        Args:
            sample_dir (str): Path to the CFDImage.mat file.
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (C, H, W).
        """
        cfd_path = os.path.join(sample_dir, "CFDImage.mat")
        cfd_mat = sio.loadmat(cfd_path)
        image_array = cfd_mat["CFDImage"].astype(np.float32)
        # image_array = cfd_mat["CFDImageOut"].astype(np.float32)
         # Check if the image should be flipped
        if self.if_needsFlipud(image_array): #if image of flame should be flipped
            if self.config.MODE != "Train" and self.config.MODE != "TrainSaveAllData":
                self.logger.warning(f"Image at {sample_dir} is right side up, flipping it.")
            
            self.config.isImgFlipped = True
            image_array = np.flipud(image_array)  # Flip the image array vertically
        image_array[image_array < self.config.setImgValZero] = 0.0 #negative values are not relevant and are set to 0.0
        #normelize
        # image_array = image_array/4095.0
        image_array = (image_array-self.config.global_img_min)/max((self.config.global_img_max-self.config.global_img_min), 1e-6)  # Avoid division by zero
        #padding
        # image = np.pad(image_array,((0,self.config.input_shape[1]-image_array.shape[0]),(0,self.config.input_shape[2]-image_array.shape[1]),(0,0)), mode='constant', constant_values=0)
        image = self.pad_or_crop_to_shape_img(image_array, self.config.input_shape)  # Pad or crop to the desired shape
       

        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert("RGB")                
        image = transforms.ToTensor()(image)
        return image

    def __getitem__(self, idx):      
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, target) where image is a tensor of shape (C, H, W) and target is a tensor of shape (2, H, W) or (H, W).
        """  
        sample_dir = self.sample_dirs[idx]
        
        image = self._getImage_(sample_dir)       

        soot_path = os.path.join(sample_dir, "sootCalculation.mat")
        soot_mat = sio.loadmat(soot_path)
        if self.config.targetType == "T":
            T = self._getT_(soot_mat)
            target = torch.tensor(T, dtype=torch.float32)
        elif self.config.targetType == "fv":
            fv = self._getFv_(soot_mat)
            target = torch.tensor(fv, dtype=torch.float32)
        elif self.config.targetType == "both":
            fv = self._getFv_(soot_mat)
            T = self._getT_(soot_mat)
            target = np.stack([fv, T], axis=0)  
            target = torch.tensor(target, dtype=torch.float32)        
        
        return image, target
       
        
