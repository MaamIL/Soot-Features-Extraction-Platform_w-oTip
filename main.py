#This file holds the configurations regarding the data creation and the model selection.
#It also holds the main function that runs the training or inference of the model.

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import scipy.io as sio
from importlib import import_module
import torch.nn.functional as F
from Logger import CustomLogger
from DataCreation import FlameDataset
from Plot_Outputs import saveheatmaps, save_error_heatmaps, save_inputImages
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuration
class Config:
    """
    Configuration class for dataset parameters, model training, and output settings.
    """    
    def __init__(self):
        """
        Initialize configuration settings.
        """
        #Params for dataset creation    
        self.paramsType2 = "~~~~~Params for dataset creation~~~~~"
        self.root_dir = 'C:/Users/User/Documents/GenerateData/OneDrive_1_04-02-2026' #GeneratedData_Inference'  # Path to your dataset GeneratedData / GeneratedData_Extra / try / GeneratedData_Inference / GeneratedData_SingleTest
        # self.modelpath = os.path.join('C:/Users/User/Documents/Sooth_Features_Extraction_plat/Train_CNNencdec_both_2025-0609-120330', 'best_flame_model.pth')
        # self.modelpath = os.path.join('C:/Users/User/Documents/Sooth_Features_Extraction_plat/TrainSaveAllData_CNNencdec_both_2025-1105-063703', 'best_flame_model.pth') #best model with tip
        self.modelpath = os.path.join('C:/Users/User/Documents/Sooth_Features_Extraction_plat/best_model_no_tip', 'best_flame_model.pth')
        self.MODE = "Inference"  # Set to "Train" or "SingleTest" or "Inference" or "TrainSaveAllData" MODE as needed (train- train the model, test- load and test the model on a single sample (input-output), inference- load model and run inference on a single sample (input only))
        ##Data for GeneratedData without image values>20000 or values<0
        ##on the with tip data
        # self.global_img_min = 0.0
        # self.global_img_max = 19960.539069855167
        # self.global_T_min = 299.0
        # self.global_T_max = 2375.0
        # self.global_fv_min = 0.0
        # self.global_fv_max = 10.940424107674488
        # self.Fvmax_height = 1218
        # self.Fvmax_width = 217
        # self.Imagemax_height = 1218
        # self.Imagemax_width = 217
        
        
        ##Data for GeneratedData without image values>20000 or values<0
        ##on without tip data
        self.global_img_min = 0.0
        self.global_img_max = 19941.026744724255
        self.global_T_min = 299.0
        self.global_T_max = 2828.0
        self.global_fv_min = 0.0
        self.global_fv_max = 11.224797513519933
        self.Fvmax_height = 808
        self.Fvmax_width = 213
        self.Imagemax_height = 808
        self.Imagemax_width = 213        
        
        # #For calculating the above params from the dataset:
        # self.global_img_min = min([
        #     sio.loadmat(os.path.join(self.root_dir, d, "CFDImage.mat"))["CFDImage"].min() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #min image value in the dataset for normalization        
        # self.global_img_max = max([
        #     sio.loadmat(os.path.join(self.root_dir, d, "CFDImage.mat"))["CFDImage"].max() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #max image value in the dataset for normalization
          
        # self.global_T_min = min([
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["T"].min() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #min temp in the dataset for normalization         
        # self.global_T_max = max([
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["T"].max() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #max temp in the dataset for normalization

        # self.global_fv_min = min([
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].min() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #min Fv in the dataset for normalization 
        # self.global_fv_max = max([
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].max() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #max Fv in the dataset for normalization  
        # #find max Fv size in dir
        # self.Fvmax_height = max(
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].shape[0]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # )
        # self.Fvmax_width = max(
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].shape[1]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # )
        # self.Imagemax_height = max(
        #     (sio.loadmat(os.path.join(self.root_dir, d, "CFDImage.mat"))["CFDImage"].shape[0]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)))
        # )
        # self.Imagemax_width = max(
        #     (sio.loadmat(os.path.join(self.root_dir, d, "CFDImage.mat"))["CFDImage"].shape[1]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)))
        # )       

        self.input_shape = (3, self.Imagemax_height, self.Imagemax_width)  # (C, H, W) for input RGB flame images
        self.output_shape = (self.Fvmax_height, self.Fvmax_width) # (Height, Width) for temperature maps
        self.targetType = "both" # "T", "fv", or "both"
        self.isNorm = True # True/False - Normalize the input images
        self.setImgValZero = 0 #50 #Set values smaller than 50 to 0 in CFDImage
        self.setFvValZero = 0.01 #Set values smaller than 0.01 to 0 in sootCalculation["fv"]
        self.setTValZero = 1000.0 #Set values smaller than 1000 to 300.0 in sootCalculation["T"]
        self.isImgFlipped = False # DO NOT CHANGE! This is used by the system which indicates if the image is upside down or not, so it will know how to handle it and how to plot it.
    # Params for model training
        self.paramsType3 = "~~~~~Params for model training~~~~~_No crop of bottom"
        self.model_name = "CNNencdec"#"CNNencdec_batch32" #"TwoStageTraining" / "MultiTaskResNet" / "CNNencdec" 
        self.batch_size = 8 # Batch size for training
        self.criterion = nn.MSELoss()
        self.lr=0.0001
        self.num_epochs = 150
        self.epochs_remark = "Patience of 10 epochs" 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        self.optimizer = "torch.optim.Adam(self.parameters(), lr=self.config.lr)"
        self.scheduler = "torch.optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.3, patience=4) "
    #Params for outputs and logging
        self.paramsType1 = "~~~~~Params for outputs and logging~~~~~"     
        self.out_dir = f'{self.MODE}_{self.model_name}_{self.targetType}_{time.strftime("%Y-%m%d-%H%M%S")}' #Path to save outputs   
        os.makedirs(self.out_dir, exist_ok=True) # Create output directory    
        self.log_filename = os.path.join(self.out_dir, "log.txt")
        self.logger = CustomLogger(self.log_filename, self.__class__.__name__).get_logger()
        self.savePlots = True #True/False - Save plots of the training process or show them without saving

    def print_config(self):
        """
        Print the configuration settings.
        """
        self.logger.info(f"""Logging to {self.log_filename}
                                          Log Format: {config.logger.logger.handlers[1].formatter._fmt}""")
        
        txt = f"\n\n~~~~~~~~~~~~~~Configuration settings~~~~~~~~~~~~~~~~~\n"
        for attr, value in self.__dict__.items():
            txt += f"{attr} = {value}\n"
        txt += f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        self.logger.info(txt)   

############################################################Functions

def prepare_data(config):
    """
    Prepare and split the data for training
    
    Args:
        config (Config): Configuration object containing dataset parameters.
    
    Returns:
        train_loader, val_loader, test_loader
    """    
    dataset = FlameDataset(config)
    
    train_size = int(0.7 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    config.logger.info(f"~~~~~~~~~~~~\nDataset sizes: \nTrain: {len(train_dataset)} \nValidation: {len(val_dataset)} \nTest: {len(test_dataset)}\n~~~~~~~~~~~~~~~~")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size+4)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    return train_loader, val_loader, test_loader


############################################################
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    #config
    config = Config()
    config.print_config()
    # Create a logger for main
    main_logger = CustomLogger(config.log_filename, __name__).get_logger()
    
    if config.MODE == "Train":
        main_logger.info("Running in training mode...")
        
    #1. Create Data Loaders
        main_logger.info("Creating dataset...")
        # Prepare data
        train_loader, val_loader, test_loader = prepare_data(config)
    #2. Run Model            
        main_logger.info("Running model...")
        try:
            model_module = import_module(f"Mymodels.{config.model_name}")
            model_class = getattr(model_module, config.model_name)
            model = model_class(config).to(config.device)

            train_losses, val_losses, test_loss, best_model = model.train_model(train_loader, val_loader, test_loader)
            main_logger.info(f"Model '{config.model_name}' trained successfully.")
            main_logger.info(f"Best model parameters:\n~~~~~~~~~~~~~~~\n{best_model.parameters()}\n saved to {os.path.join(config.out_dir, 'best_model.pth')}")
            
        except Exception as e:
            main_logger.error(f"Error loading model '{config.model_name}': {e}")
            raise
    #3. Save outputs
        try:
            model.plotLosses(train_losses, val_losses, test_loss)
            main_logger.info(f"Loss plots saved to {os.path.join(config.out_dir, 'losses.png')}")
        except Exception as e:
            main_logger.error(f"Error in saving loss plots: {e}")
    
    elif config.MODE == "TrainSaveAllData":
        main_logger.info("Running in TrainSaveAllData mode...")    
        #Create Data Loaders
        main_logger.info("Creating dataset...")
        # Prepare data
        train_loader, val_loader, test_loader = prepare_data(config)                      
        # Load model
        try:
            model_module = import_module(f"Mymodels.{config.model_name}")
            model_class = getattr(model_module, config.model_name)
            model = model_class(config).to(config.device)
            # model_path = config.modelpath
            # model.load_state_dict(torch.load(model_path, map_location=config.device), strict=False)
            # model.eval()
            # main_logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            main_logger.error(f"Error loading model: {e}")
            raise
        # Run Model
        main_logger.info("Training model...")
        try:    
            train_losses, val_losses, test_loss, best_model = model.train_model(train_loader, val_loader, test_loader)
            main_logger.info(f"Model '{config.model_name}' trained successfully.")
            main_logger.info(f"Best model parameters:\n~~~~~~~~~~~~~~~\n{best_model.parameters()}\n saved to {os.path.join(config.out_dir, 'best_model.pth')}")
            
        except Exception as e:
            main_logger.error(f"Error loading model '{config.model_name}': {e}")
            raise
    #3. Save outputs
        try:
            model.plotLosses(train_losses, val_losses, test_loss)
            main_logger.info(f"Loss plots saved to {os.path.join(config.out_dir, 'losses.png')}")
        except Exception as e:
            main_logger.error(f"Error in saving loss plots: {e}")

    elif config.MODE == "SingleTest":
        main_logger.info("Running in test mode...")    
                      
        # Load model
        try:
            model_module = import_module(f"Mymodels.{config.model_name}")
            model_class = getattr(model_module, config.model_name)
            model = model_class(config).to(config.device)
            model_path = config.modelpath
            model.load_state_dict(torch.load(model_path, map_location=config.device), strict=False)
            model.eval()
            main_logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            main_logger.error(f"Error loading model: {e}")
            raise

         # Create dataset and get 1 sample
        dataset = FlameDataset(config)
        image_tensor, gt_tensor = dataset[0]  
        image_tensor = image_tensor.unsqueeze(0).to(config.device)  # add batch dimension
        gt_tensor = gt_tensor.unsqueeze(0).to(config.device)

        # for both only
        with torch.no_grad():
            output = model(image_tensor)        
        
        if config.targetType == "both": 
            normalized_setFvValZero = (config.setFvValZero - config.global_fv_min) / max((config.global_fv_max - config.global_fv_min), 1e-6)
            normalized_setTValZero = (config.setTValZero - config.global_T_min) / max((config.global_T_max - config.global_T_min), 1e-6)
            output[:, 0, :, :][output[:, 0, :, :] < normalized_setFvValZero] = 0.0
            output[:, 1, :, :][output[:, 1, :, :] < normalized_setTValZero] = 0.0
            loss_fv = F.mse_loss(output[:, 0, :, :], gt_tensor[:, 0, :, :])
            loss_T = F.mse_loss(output[:, 1, :, :], gt_tensor[:, 1, :, :])
            loss = 0.5 * loss_fv + 0.5 * loss_T
            main_logger.info(f"Losses - fv: {loss_fv.item()}, T: {loss_T.item()}, Combined: {loss.item()}")       

            # === Save heatmaps using custom Plot_Outputs ===
        # save_dir = os.path.join(config.out_dir, "inference_results")
        os.makedirs(config.out_dir, exist_ok=True)

        # Wrap tensors back to mimic training loop shapes        
        output_tensor = output.detach().cpu()
      
        # Input already loaded as image_tensor earlier
        sample_path = dataset.sample_dirs[0]  # assuming single sample
        sample_id = os.path.splitext(os.path.basename(sample_path))[0]

        # Save heatmaps
        saveheatmaps(
            outputs=output_tensor,
            gts=gt_tensor,
            epoch="TestSingle",
            sample_number=sample_id,
            inputs=image_tensor,
            heat_dir=config.out_dir,
            sample_dir=sample_path,
            config=config
        )
 
        # Save error heatmaps
        save_error_heatmaps(
            outputs=output_tensor,
            gts=gt_tensor,
            epoch="TestSingle",
            sample_id=sample_id,
            inputs=image_tensor,
            out_dir=config.out_dir,
            sample_name=sample_path,
            config=config,
            loss=loss.item(),
            loss_fv=loss_fv.item(),
            loss_T=loss_T.item()
        )

        main_logger.info(f"TestSingle heatmaps and error maps saved to: {config.out_dir}")
    
    elif config.MODE == "Inference":
        main_logger.info("Running in inference mode...")        
                      
        # Load model
        try:
            model_module = import_module(f"Mymodels.{config.model_name}")
            model_class = getattr(model_module, config.model_name)
            model = model_class(config).to(config.device)
            model_path = config.modelpath
            model.load_state_dict(torch.load(model_path, map_location=config.device), strict=False)
            model.eval()
            main_logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            main_logger.error(f"Error loading model: {e}")
            raise
        # Load image from mat file
         # Create dataset and get 1 sample
        dataset = FlameDataset(config)
        sample_path = dataset.sample_dirs[0]  # just one sample for inference
        image_tensor = dataset._getImage_(sample_path)  # only image, no target
        image_tensor = image_tensor.unsqueeze(0).to(config.device)  # add batch dimension
        

        # Inference for both only
        with torch.no_grad():
            output = model(image_tensor)
        
        normalized_setFvValZero = (config.setFvValZero - config.global_fv_min) / max((config.global_fv_max - config.global_fv_min), 1e-6)
        normalized_setTValZero = (config.setTValZero - config.global_T_min) / max((config.global_T_max - config.global_T_min), 1e-6)
        output[:, 0, :, :][output[:, 0, :, :] < normalized_setFvValZero] = 0.0
        output[:, 1, :, :][output[:, 1, :, :] < normalized_setTValZero] = 0.0
        # save_dir = os.path.join(config.out_dir, "inference_results")
        os.makedirs(config.out_dir, exist_ok=True)

        # Wrap tensors back to mimic training loop shapes        
        output_tensor = output.detach().cpu()
      
        # Input already loaded as image_tensor earlier
        # sample_path = dataset.sample_dirs[0]  # assuming single sample
        sample_id = os.path.splitext(os.path.basename(sample_path))[0]
        # Save input image
        save_inputImages(
            inputs=image_tensor[0].cpu().detach(),
            sample_number=sample_id,
            heat_dir=config.out_dir,
            samp_folder=sample_path[sample_path.rfind('\\')+1:],
            isImgFlipped = config.isImgFlipped, 
            rootdir = config.root_dir
        )
        # Save heatmaps
        saveheatmaps(
            outputs=output,
            gts=None,
            epoch="Inference",
            sample_number=sample_id,
            inputs=image_tensor,
            heat_dir=config.out_dir,
            sample_dir=sample_path,
            config=config
        )
    

    



