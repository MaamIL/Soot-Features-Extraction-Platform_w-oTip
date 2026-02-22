import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def heatmaps(r, z, fvORt, cbar_label, pltTitle, savefile):
    """
    Create and save a heatmap of fvORt.
    This is a help function called from saveheatmaps to visualize the fv or T values.
    It generates a contour plot with a colorbar and saves it to the specified file.
    Parameters:
        r: 1D array of radial positions
        z: 1D array of axial positions
        fvORt: 2D array of fv or T values
        cbar_label: Label for the colorbar
        pltTitle: Title for the plot
        savefile: Path to save the heatmap image
    """
    # Ensure r and z are 1D arrays matching the dimensions of fvORt
    r = np.linspace(0, 1, fvORt.shape[1]) if np.isscalar(r) or len(r) != fvORt.shape[1] else r
    z = np.linspace(0, 1, fvORt.shape[0]) if np.isscalar(z) or len(z) != fvORt.shape[0] else z
    R, Z = np.meshgrid(r, z)
    plt.figure(figsize=(16, 12))
    levels = np.linspace(0, 1, 51)  # 50 intervals between 0 and 1
    contour = plt.contourf(R, Z, fvORt, levels=levels, cmap='inferno', vmin=0, vmax=1)
    plt.gca().set_aspect('equal')
    cbar = plt.colorbar(contour)
    cbar.set_label(cbar_label)
    plt.xlabel("Radial Position (r) [mm]")
    plt.ylabel("Axial Position (z) [mm]")
    plt.title(pltTitle)
    plt.savefig(savefile)
    plt.close()
from scipy.io import loadmat

def save_inputImages(inputs, sample_number, heat_dir, samp_folder, isImgFlipped, rootdir):
    # Save input image        
        if inputs.ndimension() == 3 and inputs.shape[0] == 3:           
            image_array = inputs.cpu().detach().numpy().astype(np.float32)
            image_array = np.transpose(image_array, (1, 2, 0))  # (H, W, C)
            # Normalize to [0, 1] and scale to [0, 255]
            image_array = image_array / np.max(image_array)
            image_uint8 = (image_array * 255).astype(np.uint8)
            # if not isImgFlipped:  # if image of flame was flipped - flip back
            #     image_uint8 = np.flipud(image_uint8)
            # Convert to PIL image and save
            image = Image.fromarray(image_uint8).convert("RGB")
            image.save(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Input.jpg'))
    #save original image (might be different due to flips, crop to size, etc.)
        cfd_path = os.path.join(rootdir, samp_folder, "CFDImage.mat")
        cfd_mat = loadmat(cfd_path)
        image_array = cfd_mat["CFDImage"].astype(np.float32)
        # image_array = cfd_mat["CFDImageOut"].astype(np.float32)        
        # Normalize to [0, 1] and scale to [0, 255]
        image_array = image_array / np.max(image_array)
        image_uint8 = (image_array * 255).astype(np.uint8)
        # if not isImgFlipped:  # if image of flame was flipped - flip back
        #     image_uint8 = np.flipud(image_uint8)
        # Convert to PIL image and save
        image = Image.fromarray(image_uint8).convert("RGB")
        image.save(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_OrigImage.jpg'))

def save_csv(filename, data):
    """Save data to a CSV file.
    Parameters:
        filename: str - path to the CSV file
        data: np.ndarray - data to save, should be 2D array-like
    """
    np.savetxt(filename, data, delimiter=",", fmt="%.6f")

def saveheatmaps(outputs, gts, epoch, sample_number, inputs, heat_dir, sample_dir, config):
    """
    Saves heatmaps of fv and T predictions and ground truths. Saves also input image.
    Parameters:
        outputs: (B, 2, H, W) - model predictions
        gts: (B, 2, H, W) - ground truth maps
        epoch: int or str - current epoch or "Test"/"TestSingle"
        sample_number: str - index or tag for the sample
        inputs: (B, C, H, W) - original input images
        heat_dir: str - directory to save heatmaps
        sample_dir: str - directory of the sample
        config: config object with normalization info if needed
    """
    samp_folder = sample_dir[sample_dir.rfind('\\')+1:]

    # Convert tensors to CPU and detach for processing
    preds = outputs[0].cpu().detach()
    if gts not in [None]:
         gts = gts[0].cpu().detach() 
    inputs = inputs[0].cpu().detach()
    
    # Optional image save (only once, epoch=0 or test/TestSingle sample)
    if (epoch == 0) or (epoch == "Test") or (epoch == "TestSingle"):
        # Save input image        
        save_inputImages(inputs, sample_number, heat_dir, samp_folder, config.isImgFlipped, config.root_dir)

        # Save GT heatmaps
        if gts.shape[0] == 2: # Assuming we are in both mode- gts has T and Fv
            fv_gt = gts[0].numpy()
            T_gt = gts[1].numpy()
            save_csv(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Fv_GT.csv'), fv_gt)
            save_csv(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_T_GT.csv'), T_gt)
            for arr, name, cbar in zip([fv_gt, T_gt], ["Fv_GT", "T_GT"], ["$Fv(r, z)$ [ppm]", "$T(r, z)$ [K]"]):
                r = np.linspace(0, 1, arr.shape[1])
                z = np.linspace(0, 1, arr.shape[0])
                title = f"{sample_number}_Heatmap of {name} ({sample_dir})"
                savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_{name}.jpg')
                heatmaps(r, z, arr, cbar, title, savefile)

        else: # Assuming we are in single mode- gts has T or Fv
            arr_gt = gts.numpy()
            r = np.linspace(0, 1, arr_gt.shape[1])
            z = np.linspace(0, 1, arr_gt.shape[0])
            if config.targetType == "T":
                 title = f"{sample_number}_Heatmap of T_GT ({sample_dir})"
                 cbarTitle = '$T(r, z)$ [K]'
                 savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_T_GT.jpg')
            elif config.targetType == "fv":
                title = f"{sample_number}_Heatmap of Fv_GT ({sample_dir})"
                cbarTitle = '$Fv(r, z)$ [ppm]'
                savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Fv_GT.jpg')
            heatmaps(r, z, arr_gt, cbarTitle, title, savefile)

    # Save predicted heatmaps
    if preds.shape[0] == 2: # Assuming we are in both mode- gts has T and Fv
        fv_pred = preds[0].numpy()
        T_pred = preds[1].numpy()
        if (epoch == "Test") or (epoch == "TestSingle") or (epoch == "Inference"):
            save_csv(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Fv_Pred.csv'), fv_pred)
            save_csv(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_T_Pred.csv'), T_pred)
        for arr, name, cbar in zip([fv_pred, T_pred], ["Fv_Pred", "T_Pred"], ["$Fv(r, z)$ [ppm]", "$T(r, z)$ [K]"]):
            r = np.linspace(0, 1, arr.shape[1])
            z = np.linspace(0, 1, arr.shape[0])
            title = f"{sample_number}_Heatmap of {name} Epoch {epoch} ({sample_dir})"
            savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}_{name}.jpg')
            heatmaps(r, z, arr, cbar, title, savefile)
    else: # Assuming we are in single mode- gts has T or Fv
        arr_pred = preds[0].numpy()
        r = np.linspace(0, 1, arr_pred.shape[1])
        z = np.linspace(0, 1, arr_pred.shape[0])
        if config.targetType == "T":
            title = f"{sample_number}_Heatmap of T_Pred Epoch {epoch} ({sample_dir})"
            cbarTitle = '$T(r, z)$ [K]'
            savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}_T_Pred.jpg')
        elif config.targetType == "fv":
            title = f"{sample_number}_Heatmap of Fv_Pred Epoch {epoch} ({sample_dir})"
            cbarTitle = '$Fv(r, z)$ [ppm]'
            savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}_Fv_Pred.jpg')
        heatmaps(r, z, arr_pred, cbarTitle, title, savefile)

def save_error_heatmaps(outputs, gts, epoch, sample_id, inputs, out_dir, sample_name, config, loss, loss_fv, loss_T):
    """
    Saves per-pixel absolute error heatmaps for fv and T - prediction compared to GT.
    Parameters:
        outputs: (B, 2, H, W) - model predictions
        gts: (B, 2, H, W) - ground truth maps
        epoch: int - current epoch
        sample_id: str - index or tag for the sample
        inputs: (B, C, H, W) - original input images
        out_dir: str - base output directory
        sample_name: str - name or path of the sample
        config: config object with normalization info if needed
        loss: float - overall loss for the sample
        loss_fv: float - loss for fv predictions
        loss_T: float - loss for T predictions
    """
    outputs = outputs.detach().cpu().numpy()
    gts = gts.detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()

    batch_size = outputs.shape[0]
    for i in range(batch_size):
        pred = outputs[i]
        gt = gts[i]

        if config.targetType == "both":
            error_fv = np.abs(pred[0] - gt[0])
            error_T = np.abs(pred[1] - gt[1])
        elif config.targetType == "fv":
            error_fv = np.abs(pred[0] - gt[0])
            error_T = None
        elif config.targetType == "T":
            error_T = np.abs(pred[0] - gt[0])
            error_fv = None
        
        fig, axs = plt.subplots(2, 2, figsize=(7, 10))
        # Set the figure title
        fig.suptitle(f"Sample: {sample_id} - Epoch: {epoch} - Loss: {loss:.8f}\n", fontsize=14, color='darkred')
        # FV error heatmaps (one with fixed scale of 0-1, one with dynamic scale, so it will be easier to see the differences)
        if config.targetType in ["both", "fv"] and error_fv is not None:
            im1 = axs[1][0].imshow(np.flipud(error_fv), cmap='hot', vmin=0, vmax=np.max(error_fv) if np.max(error_fv) > 0 else 1)
            axs[1][0].set_title(f"Error Heatmap - fv - Loss: {loss_fv:.8f}")
            fig.colorbar(im1, ax=axs[1][0])
            im2 = axs[1][1].imshow(np.flipud(error_fv), cmap='hot', vmin=0, vmax=1)
            axs[1][1].set_title("(fixed scale)")
            fig.colorbar(im2, ax=axs[1][1])
        else:
            axs[1][0].axis('off')
            axs[1][1].axis('off')

        # T error heatmaps (one with fixed scale of 0-1, one with dynamic scale, so it will be easier to see the differences)
        if config.targetType in ["both", "T"] and error_T is not None:
            im3 = axs[0][0].imshow(np.flipud(error_T), cmap='hot', vmin=0, vmax=np.max(error_T) if np.max(error_T) > 0 else 1)
            axs[0][0].set_title(f"Error Heatmap - T - Loss: {loss_T:.8f}")
            fig.colorbar(im3, ax=axs[0][0])
            im4 = axs[0][1].imshow(np.flipud(error_T), cmap='hot', vmin=0, vmax=1)
            axs[0][1].set_title("(fixed scale)")
            fig.colorbar(im4, ax=axs[0][1])
        else:
            axs[0][0].axis('off')
            axs[0][1].axis('off')

        for row in axs:
            for ax in row:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_id}_{os.path.basename(sample_name)}_E{epoch}_ErrorMaps.png"))
        plt.close()
        