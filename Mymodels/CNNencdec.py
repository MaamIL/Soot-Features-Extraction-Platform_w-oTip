import numpy as np
from Logger import CustomLogger
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from Plot_Outputs import saveheatmaps, save_error_heatmaps
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CNNencdec(nn.Module):
    """
    CNN Encoder-Decoder model for predicting flame properties from CFD images.
    """
    def __init__(self, config):
        """
        Initialize the CNN Encoder-Decoder model.
        Args:
            config (object): Configuration object containing model and data parameters.
        """
         # Initialize the logger
        super(CNNencdec, self).__init__()
        self.config = config        
        self.logger = CustomLogger(self.config.log_filename, self.__class__.__name__).get_logger()
        #Model's layers
        in_channels = self.config.input_shape[0]
        out_channels = 2 if self.config.targetType == "both" else 1
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        #Model's configurations
        self.config.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
          # scheduler- reduce lr in case of plateau in validation loss (no improvements after *patience* epochs), then learning rate will be reduced: new_lr = lr * factor
        self.config.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.3, patience=3) 
        # self.config.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.config.optimizer, T_max=self.config.num_epochs, eta_min=0.000001)
        self.logger.info(f"""
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Model '{config.model_name}' initialized on {config.device}
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        enc_out, features = self.encoder(x)
        dec_out = self.decoder(enc_out, features)
        out = self.final(dec_out)
        out = torch.sigmoid(out)
        # Match output to desired size (e.g. 808x213)
        target_h, target_w = self.config.output_shape
        if out.shape[2] != target_h or out.shape[3] != target_w:
            out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return out

    def save_predictions_from_loader(self, loader, dataset_name):
        # self.eval()
        # self.load_state_dict(torch.load(os.path.join(self.config.out_dir, "best_flame_model.pth")))
        csvdir = os.path.join(self.config.out_dir, f"csv_data_best_{dataset_name}")
        os.makedirs(csvdir, exist_ok=True)

        self.eval()
        with torch.no_grad():
            for batch_i, (inputs, gts) in enumerate(tqdm(loader, desc=f"Saving {dataset_name} predictions")):
                inputs = inputs.to(self.config.device)
                gts = gts.to(self.config.device)
                outputs = self(inputs)

                # Get base dataset and indices
                dataset = loader.dataset
                if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                    base_dataset = dataset.dataset
                    indices = dataset.indices
                else:
                    base_dataset = dataset
                    indices = list(range(len(dataset)))

                batch_size = inputs.shape[0]

                for j in range(batch_size):
                    sample_idx = indices[batch_i * loader.batch_size + j]
                    sample_name = os.path.basename(base_dataset.sample_dirs[sample_idx])

                    gt = gts[j].detach().cpu().numpy()
                    output = outputs[j].detach().cpu().numpy()

                    np.savetxt(os.path.join(csvdir, f"{sample_name}_gt_Fv_{dataset_name}.csv"), gt[0], delimiter=",", fmt="%.6f")
                    np.savetxt(os.path.join(csvdir, f"{sample_name}_gt_T_{dataset_name}.csv"), gt[1], delimiter=",", fmt="%.6f")
                    np.savetxt(os.path.join(csvdir, f"{sample_name}_pred_Fv_{dataset_name}.csv"), output[0], delimiter=",", fmt="%.6f")
                    np.savetxt(os.path.join(csvdir, f"{sample_name}_pred_T_{dataset_name}.csv"), output[1], delimiter=",", fmt="%.6f")


    def train_model(self, train_loader, val_loader, test_loader):
        """
        Train the model using the provided data loaders.
        Each epoch consists of training and validation phases.
        The model is evaluated on the test set after training.
        During validation and testing, the model saves heatmaps of predictions and ground truths for visual inspection.
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
            Returns:
            tuple: Lists of training losses, validation losses, and test loss.
        """
        self.to(self.config.device)
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        print("\n\n##############Train batch size:", train_loader.batch_size, "\n##############Validation batch size:", val_loader.batch_size, "\n##############Test batch size:", test_loader.batch_size)

        for epoch in range(self.config.num_epochs):
            self.train()
            running_loss = 0.0
            #use tqdm for progress bar
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]") as pbar:
                # for inputs, gts in pbar:
                for i, (inputs, gts) in enumerate(pbar):
                    inputs = inputs.to(self.config.device)
                    gts = gts.to(self.config.device)

                    self.config.optimizer.zero_grad()
                    outputs = self(inputs)  
                    #calculate losses 
                    if self.config.targetType == "both":
                        loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                        loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                        loss = 0.5 * loss_fv + 0.5 * loss_T
                    elif self.config.targetType == "fv":
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    elif self.config.targetType == "T":
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    
                    loss.backward()
                    self.config.optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})   
                    # #save csv files (done for retreiving data for plotting) only for mode "TrainSaveAllData"                
                    # if self.config.MODE == "TrainSaveAllData":
                    #     
                    #     csvdir = os.path.join(self.config.out_dir, "csv_data")
                    #     sample_name = os.path.basename(train_loader.dataset.dataset.sample_dirs[train_loader.sampler.data_source.indices[i]])
                    #     if epoch == 0:
                    #         os.makedirs(self.config.out_dir, exist_ok=True)                            
                    #         os.makedirs(csvdir, exist_ok=True)   
                    #         for gt in gts:                                
                    #             gt = gt.detach().cpu().numpy()
                    #             filename = os.path.join(csvdir, f"{sample_name}_gt_Fv_train.csv")
                    #             np.savetxt(filename, gt[0], delimiter=",", fmt="%.6f")
                    #             filename = os.path.join(csvdir, f"{sample_name}_gt_T_train.csv")
                    #             np.savetxt(filename, gt[1], delimiter=",", fmt="%.6f")
                    #     for output in outputs:                                
                    #         output = output.detach().cpu().numpy()
                    #         filename = os.path.join(csvdir, f"{sample_name}_pred_Fv_train.csv")
                    #         np.savetxt(filename, output[0], delimiter=",", fmt="%.6f")
                    #         filename = os.path.join(csvdir, f"{sample_name}_pred_T_train.csv")
                    #         np.savetxt(filename, output[1], delimiter=",", fmt="%.6f")

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            #validation phase
            self.eval()
            val_loss = 0.0
            print4samples = 0 # Number of samples to print heatmaps for
            early_stop_patience = 10#15 # Number of epochs to wait before early stopping
            if epoch == 5:
                print("epoch 5- Setting BatchNorm layers to eval mode during validation/testing")
                for m in self.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Val]") as pbar:
                    for i, (inputs, gts) in enumerate(pbar):
                        inputs = inputs.to(self.config.device)
                        gts = gts.to(self.config.device)

                        outputs = self(inputs) 

                        
                        # print(f"Val batch {i}: {inputs.size(0)}")
                         
                        #calculate losses
                        if self.config.targetType == "both":
                            loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                            loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                            loss = 0.5 * loss_fv + 0.5 * loss_T
                        elif self.config.targetType == "fv":
                            loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                        elif self.config.targetType == "T":
                            loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                        val_loss += loss.item()
                        pbar.set_postfix({"loss": loss.item()})
                        #
                        # if self.config.MODE == "TrainSaveAllData":
                        #     import numpy as np
                        #     csvdir = os.path.join(self.config.out_dir, "csv_data")
                        #     sample_name = os.path.basename(val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]])
                        #     # if epoch == 0:
                        #     os.makedirs(self.config.out_dir, exist_ok=True)                            
                        #     os.makedirs(csvdir, exist_ok=True)   
                        #     for gt in gts:                                
                        #         gt = gt.detach().cpu().numpy()
                        #         filename = os.path.join(csvdir, f"{sample_name}_gt_Fv_val.csv")
                        #         np.savetxt(filename, gt[0], delimiter=",", fmt="%.6f")
                        #         filename = os.path.join(csvdir, f"{sample_name}_gt_T_val.csv")
                        #         np.savetxt(filename, gt[1], delimiter=",", fmt="%.6f")
                        #     for output in outputs:                                
                        #         output = output.detach().cpu().numpy()
                        #         filename = os.path.join(csvdir, f"{sample_name}_pred_Fv_val.csv")
                        #         np.savetxt(filename, output[0], delimiter=",", fmt="%.6f")
                        #         filename = os.path.join(csvdir, f"{sample_name}_pred_T_val.csv")
                        #         np.savetxt(filename, output[1], delimiter=",", fmt="%.6f")
                            
                        # Save heatmaps and error heatmaps for visual inspection for 4 samples every 10 epochs
                        if (epoch%10 == 0) and print4samples < 4 :
                            saveheatmaps(outputs, gts, epoch, str(i)+"_", inputs,
                                         self.config.out_dir,
                                         val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]],
                                         self.config)      
                            save_error_heatmaps(outputs, gts, epoch, str(i)+"_", inputs, self.config.out_dir,
                                val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]],
                                self.config, loss.item(), loss_fv.item() if self.config.targetType == "both" else None,
                                loss_T.item() if self.config.targetType == "both" else None)
                            print4samples += 1
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            self.logger.info(f"Epoch {epoch+1}, lr: {self.config.optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f} (best: {best_val_loss:.8f})")

            if hasattr(self.config, 'scheduler') and self.config.scheduler:
                self.config.scheduler.step(avg_val_loss)

            # Early stopping and save best model are based on validation loss  
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), os.path.join(self.config.out_dir, "best_flame_model.pth"))
                self.logger.info(f"Best model saved with val loss: {best_val_loss:.8f}")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1} due to no improvement in val loss for {early_stop_patience} epochs.")
                break
            torch.cuda.empty_cache()

        # Test the model
        self.logger.info(f"\n\nTesting model on Best model saved with val loss: {best_val_loss:.8f}")
        # Load the best saved model 
        self.load_state_dict(torch.load(os.path.join(self.config.out_dir, "best_flame_model.pth")))
        self.to(self.config.device)
        self.eval()
        if self.config.MODE == "TrainSaveAllData":
            self.save_predictions_from_loader(train_loader, "train")
            self.save_predictions_from_loader(val_loader, "val")
            # self.save_predictions_from_loader(test_loader, "test")
        test_loss = 0.0
        print10samples = 0 #print 10 test samples heatmaps
        with torch.no_grad():
            with tqdm(test_loader, desc="Testing") as pbar:
                for i, (inputs, gts) in enumerate(pbar):
                    inputs = inputs.to(self.config.device)
                    gts = gts.to(self.config.device)
                    outputs = self(inputs)  
                                        
                    # Calculate losses 
                    if self.config.targetType == "both":
                        normalized_setFvValZero = (self.config.setFvValZero - self.config.global_fv_min) / max((self.config.global_fv_max - self.config.global_fv_min), 1e-6)
                        normalized_setTValZero = (self.config.setTValZero - self.config.global_T_min) / max((self.config.global_T_max - self.config.global_T_min), 1e-6)
                        outputs[:, 0, :, :][outputs[:, 0, :, :] < normalized_setFvValZero] = 0.0
                        outputs[:, 1, :, :][outputs[:, 1, :, :] < normalized_setTValZero] = 0.0
                        loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                        loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                        loss = 0.5 * loss_fv + 0.5 * loss_T
                    elif self.config.targetType == "fv":
                        normalized_setFvValZero = (self.config.setFvValZero - self.config.global_fv_min) / max((self.config.global_fv_max - self.config.global_fv_min), 1e-6)
                        outputs[:, 0, :, :][outputs[:, 0, :, :] < normalized_setFvValZero] = 0.0
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    elif self.config.targetType == "T":
                        normalized_setTValZero = (self.config.setTValZero - self.config.global_T_min) / max((self.config.global_T_max - self.config.global_T_min), 1e-6)
                        outputs[:, 0, :, :][outputs[:, 0, :, :] < normalized_setTValZero] = 0.0
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    test_loss += loss.item()
                    #
                    if self.config.MODE == "TrainSaveAllData":
                        # import numpy as np
                        csvdir = os.path.join(self.config.out_dir, "csv_data_best_test")
                        sample_name = os.path.basename(test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]])                        
                        os.makedirs(self.config.out_dir, exist_ok=True)                            
                        os.makedirs(csvdir, exist_ok=True)   
                        for gt in gts:                                
                            gt = gt.detach().cpu().numpy()
                            filename = os.path.join(csvdir, f"{sample_name}_gt_Fv_test.csv")
                            np.savetxt(filename, gt[0], delimiter=",", fmt="%.6f")
                            filename = os.path.join(csvdir, f"{sample_name}_gt_T_test.csv")
                            np.savetxt(filename, gt[1], delimiter=",", fmt="%.6f")
                        for output in outputs:                                
                            output = output.detach().cpu().numpy()
                            filename = os.path.join(csvdir, f"{sample_name}_pred_Fv_test.csv")
                            np.savetxt(filename, output[0], delimiter=",", fmt="%.6f")
                            filename = os.path.join(csvdir, f"{sample_name}_pred_T_test.csv")
                            np.savetxt(filename, output[1], delimiter=",", fmt="%.6f")
                    # Save heatmaps and error heatmaps for visual inspection for 10 samples
                    if print10samples < 10:
                        saveheatmaps(outputs, gts, "Test", str(i)+"_", inputs,
                                    self.config.out_dir,
                                    test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]],
                                    self.config)
                        save_error_heatmaps(outputs, gts, "Test", str(i)+"_", inputs, self.config.out_dir,
                                    test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]],
                                    self.config, loss.item(), loss_fv.item() if self.config.targetType == "both" else None,
                                    loss_T.item() if self.config.targetType == "both" else None)
                        
                        print10samples += 1
        test_loss /= len(test_loader)        
        self.logger.info(f"Test Loss: {test_loss:.8f}")
        #plot losses 
        self.plotLosses(train_losses, val_losses, test_loss)
        return train_losses, val_losses, test_loss, self

    def plotLosses(self, train_losses, val_losses, test_loss):
        """
        Plot training and validation losses, and save the plots.
        Args:
            train_losses (list): List of training losses per epoch.
            val_losses (list): List of validation losses per epoch.
            test_loss (float): Final test loss after training.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.axhline(test_loss, color='red', linestyle='--', label=f'Test Loss: {test_loss:.8f}')
        plt.title("Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.out_dir, "losses.png"))
        plt.close()

        # Same plots, Zoomed in on the y-axis
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.axhline(test_loss, color='red', linestyle='--', label=f'Test Loss: {test_loss:.8f}')
        plt.title("Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.ylim(0, 0.001)  
        plt.savefig(os.path.join(self.config.out_dir, "losses_zoom.png"))
        plt.close()
        

### NETWORK MODULES ###

class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and a skip connection.
    Used in the encoder part of the CNN.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the ResidualBlock.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolutional layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the residual block. Add skip connection to the output of the second convolutional layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:    
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)


class Encoder(nn.Module):
    """
    CNN Encoder module that extracts features from the input image.
    It consists of an initial convolutional layer followed by a few residual blocks."""
    def __init__(self, in_channels):
        """
        Initialize the Encoder.
        Args:
            in_channels (int): Number of input channels (3 for RGB images).
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            # ResidualBlock(512, 512, stride=2),
            ## ResidualBlock(512, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )

    def forward(self, x):
        """
        Forward pass of the encoder. Pass the input through the initial layer and then through the residual blocks.
        Here, we also collect feature maps from each block for adding to the decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            tuple: Output tensor of shape (batch_size, 512, height/32, width/32) and a list of feature maps from each block.
        """
        x0 = self.initial(x)
        features = [x0]
        for block in self.blocks:
            x0 = block(x0)
            features.append(x0)
        return x0, features[::-1]


class Decoder(nn.Module):
    """
    CNN Decoder module that reconstructs the output from the encoded features.
    It consists of several upsampling blocks that progressively increase the spatial dimensions of the feature maps.
    """
    def __init__(self):
        """
        Initialize the Decoder.
        """
        super().__init__()
        self.up_blocks = nn.ModuleList([
            self._up_block(512, 512, 0.2),
            # self._up_block(512, 512, 0.3),
            self._up_block(512, 256, 0.3),
            self._up_block(256, 128, 0.2),
            self._up_block(128, 64, 0.1)
        ])

    def _up_block(self, in_ch, out_ch, dropout=0.2):
        """
        Create a single upsampling block with transposed convolution, batch normalization, ReLU activation, and dropout.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            dropout (float): Dropout probability for regularization.
        Returns:
            nn.Sequential: A sequential block containing the upsampling layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)  
        )

    def forward(self, x, features):
        """
        Forward pass of the decoder. Upsample the input feature map and add skip connections from the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 512, height/32, width/32).
            features (list): List of feature maps from the encoder.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 64, height, width) after upsampling and skip connections.
        """
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if i + 1 < len(features):
                enc_feat = features[i + 1]
                if x.shape != enc_feat.shape:
                    enc_feat = F.interpolate(enc_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + enc_feat  # Skip connection
        return x
