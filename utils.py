import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import OpenEXR
import Imath
from pathlib import Path
import re
import os

class RenderSequence:
    def __init__(self, folder_path, device="cuda"):
        self.folder_path = Path(folder_path)
        self.device = device
        self.pt = Imath.PixelType(Imath.PixelType.FLOAT)
        self.frames = {} 
        self._scan_folder()
        
    def _scan_folder(self):
        pattern = re.compile(r"(.+)_(\d{4})\.exr")
        for file in os.listdir(self.folder_path):
            match = pattern.match(file)
            if match:
                pass_name, frame_idx = match.groups()
                frame_idx = int(frame_idx)
                if frame_idx not in self.frames:
                    self.frames[frame_idx] = {}
                self.frames[frame_idx][pass_name] = self.folder_path / file
        
        self.available_indices = sorted(self.frames.keys())
        print(f"Séquence chargée : {len(self.available_indices)} frames trouvées.")

    def _load_exr(self, path):
        """Charge l'EXR en tenseur linéaire (B, C, H, W) sur GPU."""
        file = OpenEXR.InputFile(str(path))
        header = file.header()
        dw = header['displayWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # On définit explicitement les canaux pour garantir l'ordre R, G, B
        channel_names = ['R', 'G', 'B']
        tensors = []
        
        for c in channel_names:
            if c in header['channels']:
                buf = file.channel(c, self.pt)
                # Conversion rapide via numpy
                c_data = np.frombuffer(buf, dtype=np.float32).reshape(height, width)
                tensors.append(torch.from_numpy(c_data.copy()))
            else:
                # Canal manquant (ex: passe de profondeur monocanal)
                tensors.append(torch.zeros((height, width)))
        
        # On stack sur la dimension 0 (Canaux) -> (C, H, W)
        full_tensor = torch.stack(tensors)
        
        # Transfert GPU et ajout de la dimension Batch -> (1, C, H, W)
        return full_tensor.to(self.device).unsqueeze(0)

    def get_frame_pass(self, frame_idx, pass_name):
        if frame_idx not in self.frames:
            raise ValueError(f"Frame {frame_idx} non trouvée.")
        if pass_name not in self.frames[frame_idx]:
            raise ValueError(f"Passe '{pass_name}' non disponible.")
            
        return self._load_exr(self.frames[frame_idx][pass_name])

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, index):
        """Retourne toutes les passes d'une frame sous forme de dictionnaire de tenseurs."""
        return {name: self.get_frame_pass(index, name) for name in self.frames[index]}


def save_tensor_as_exr(tensor, file_path):
    """
    Save a pytorch tensor (1, 3, H, W) in EXR format
    """
    # We remove batch dimension and transform to (H, W, C)
    img = tensor.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
    
    height, width, depth = img.shape
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    
    exr = OpenEXR.OutputFile(str(file_path), header)
    
    # Separation of channels and writing
    r = img[:, :, 0].tobytes()
    g = img[:, :, 1].tobytes()
    b = img[:, :, 2].tobytes()
    
    exr.writePixels({'R': r, 'G': g, 'B': b})
    exr.close()
    print(f"File saved : {file_path}")


######### UNET #########
def preprocessing(f_tildeA, f_tildeB, textures, normals, f_prev_denoised=None):
    """
    Prépare le tenseur d'entrée pour le U-Net (15 canaux).
    
    Arguments:
        f_tildeA: (B, 3, H, W) - Estimation pilote A (HDR linéaire)
        f_tildeB: (B, 3, H, W) - Estimation pilote B (HDR linéaire)
        textures: (B, 3, H, W) - Albedo (linéaire, entre 0 et 1)
        normals:  (B, 3, H, W) - Normales (entre -1 et 1)
        f_prev_denoised: (B, 3, H, W) - Sortie finale à T-1 (HDR linéaire)
    """
    

    # Log Transform : f(x) = log(1 + x)
    # We use clamp(min=0) before the log because some pixels might be negative
    f_tildeA_log = torch.log1p(torch.clamp(f_tildeA, min=0.0))
    f_tildeB_log = torch.log1p(torch.clamp(f_tildeB, min=0.0))
    
    # Normalization of the normals
    norm_processed = F.normalize(normals, dim=1)

    # Manage previous frame
    if f_prev_denoised is None:
        # For the first frame there is no history
        # We use the mean of pilots as best estimation
        f_prev_log = (f_tildeA_log + f_tildeB_log) * 0.5
    else:
        f_prev_log = torch.log1p(torch.clamp(f_prev_denoised, min=0.0)).clone()

    # Concatenation
    # We stach them all on channel dimension (dim=1)
    # f_tildeA (3) + f_tildeB (3) + textures (3) + normals (3) + f_prev (3) 
    # = 15 channels
    input_tensor = torch.cat([
        f_tildeA_log, 
        f_tildeB_log,
        textures, 
        norm_processed, 
        f_prev_log
    ], dim=1)

    return input_tensor


class SmallUNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=6):
        super(SmallUNet, self).__init__()

        # --- Encoding (Contracting Path) ---
        self.enc1 = self.conv_block(in_channels, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2 = self.conv_block(8, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3 = self.conv_block(16, 32)
        
        # --- Décoding (Expansive Path) ---
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(32 + 16, 16) # Skip connection of enc2
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(16 + 8, 8)   # Skip connection of enc1
        
        # Out layer (1x1 Convolution)
        # Produces the 6 parameters theta per pixel
        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        
        b = self.enc3(p2)
        
        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
    

#### Loss Functions ####
def spatial_loss_fn(f_hat_A, f_hat_B, f_tilde_A, f_tilde_B, epsilon=1e-4):
    """    
    Args:
        f_hat_A: (B, C, H, W) - Denoised output branch A
        f_hat_B: (B, C, H, W) - Denoised output branch B
        f_tilde_A: (B, C, H, W) - Pilot output A
        f_tilde_B: (B, C, H, W) - Pilot output B
    """
    diff_squared_1 = (f_hat_A - f_tilde_B) ** 2
    diff_squared_2 = (f_hat_B - f_tilde_A) ** 2
    
    norm_A = f_tilde_A ** 2 + epsilon
    norm_B = f_tilde_B ** 2 + epsilon
    
    term_1 = diff_squared_1 / norm_B
    
    term_2 = diff_squared_2 / norm_A
    
    loss_per_pixel = 0.5 * (term_1 + term_2)
    
    return torch.mean(loss_per_pixel)


def prepare_motion_vectors(mvec, H, W):  
    """
    Prepare motion vectors
    
    :param mvec: Tensor of shape (B, C_mv, H, W) containing the motion vectors
    :param H: Height (in pixels)
    :param W: Width (in pixels)
    """

    # If mvec has more than 2 channels (e.g RGBA), we take the two first
    if mvec.shape[1] > 2:
        mvec = mvec[:, 0:2, :, :]

    # If max > 2.0, they might be pixels    
    if torch.max(torch.abs(mvec)) > 2.0:
        # Conversion Pixels -> Normalize [-1, 1] for grid_sample
        mvec_norm = mvec.clone()
        mvec_norm[:, 0, :, :] = (mvec[:, 0, :, :] / W) * 2.0
        mvec_norm[:, 1, :, :] = (mvec[:, 1, :, :] / H) * 2.0
        return mvec_norm
    else:
        return mvec


def warp(image, motion_vectors, mode='bilinear', padding_mode='border'):
    """
    Deform previous frame (t-1) to current frame (t)
    """
    B, C, H, W = image.shape
    device = image.device
        
    xx = torch.arange(0, W, device=device).float()
    yy = torch.arange(0, H, device=device).float()
    
    # Normalization [-1, 1]
    xx = (xx + 0.5) / W * 2.0 - 1.0 
    yy = (yy + 0.5) / H * 2.0 - 1.0
    
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    
    # (B, H, W, 2)
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Permutation (B, 2, H, W) -> (B, H, W, 2)
    mv = motion_vectors.permute(0, 2, 3, 1)
    
    sampling_grid = base_grid + mv
    
    # Sampling
    warped = F.grid_sample(image, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    
    return warped


def temporal_loss_fn(f_hat_A, f_hat_B, f_tilde_prev_A, f_tilde_prev_B, motion_vectors, epsilon=1e-4):
    """
    Implementation of temporal loss
    
    Args:
        f_hat_A, f_hat_B: (B, C, H, W) - Current output
        f_tilde_prev_A, f_tilde_prev_B: (B, C, H, W) - Pilot output of previous frame
        motion_vectors: (B, 2, H, W) - Motion vection for reprojection of t-1 to t
    """
    
    #If no previous frame, loss=0.0
    if f_tilde_prev_A is None or f_tilde_prev_B is None:
        return torch.tensor(0.0, device=f_hat_A.device)

    # Reprojection of previous frames
    warped_f_tilde_prev_A = warp(f_tilde_prev_A, motion_vectors)
    warped_f_tilde_prev_B = warp(f_tilde_prev_B, motion_vectors)
    
    diff_sq_A = (f_hat_A - warped_f_tilde_prev_B) ** 2
    norm_B_prev = warped_f_tilde_prev_B ** 2 + epsilon
    term_A = diff_sq_A / norm_B_prev
    
    diff_sq_B = (f_hat_B - warped_f_tilde_prev_A) ** 2
    norm_A_prev = warped_f_tilde_prev_A ** 2 + epsilon
    term_B = diff_sq_B / norm_A_prev
    
    loss_map = 0.5 * (term_A + term_B)
    
    return torch.mean(loss_map)


##### Spatiotemporal Filter ####

def create_position_buffer(B, H, W, device):
    """
    Generate position buffer (x, y) normalized between 0 and 1
    
    Args:
        B (int): Batch size
        H (int): Height of image
        W (int): Width of image
    """
    # Creation of linear vectors from 0 to 1
    y_coords = torch.linspace(0, 1, steps=H, device=device)
    x_coords = torch.linspace(0, 1, steps=W, device=device)
    
    # Creation of the grid (Meshgrid)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    zeros = torch.zeros_like(grid_x)
    
    # Stacking
    pos_2d = torch.stack([grid_x, grid_y, zeros], dim=0)
    
    pos_batch = pos_2d.unsqueeze(0).repeat(B, 1, 1, 1)
    
    return pos_batch


def final_spatiotemporal_pipeline(theta, f_tilde_A, f_tilde_B, textures, 
                                  normals, positions, f_prev=None,                                   
                                  kernel_size=11, tile_size=256):
    """
    Complete denoising pipeline
    """
    B, C, H, W = f_tilde_A.shape
    device = f_tilde_A.device
    pad = kernel_size // 2
    N = kernel_size**2
    epsilon = 1e-6

    # Output tensor
    final_output = torch.zeros((B, 3, H, W), device=device)

    # --- Local functions ---

    def get_patches(tensor):
        """ (B, C, H_pad, W_pad) -> (B, C, N, h_out, w_out) """
        patches = F.unfold(tensor, kernel_size=kernel_size, padding=0)
        h_out = tensor.shape[2] - 2 * pad
        w_out = tensor.shape[3] - 2 * pad
        return patches.view(B, tensor.shape[1], N, h_out, w_out)

    def center_crop(tensor):
        """ Retire le padding contextuel pour garder la zone valide """
        return tensor[:, :, pad:-pad, pad:-pad]

    def compute_weights(f_in, var_color_in, n_in, rho_in, p_in, theta_full):
        p_f, p_n = get_patches(f_in), get_patches(n_in) # 1 x 3 x 121 x 256 x 256
        p_rho, p_p = get_patches(rho_in), get_patches(p_in)
        
        c_f = center_crop(f_in).unsqueeze(2) # 1 x 3 x 1 x 256 x 256
        c_n = center_crop(n_in).unsqueeze(2)
        c_rho = center_crop(rho_in).unsqueeze(2)
        c_p = center_crop(p_in).unsqueeze(2)
        
        t_c = center_crop(theta_full)
        v_c = var_color_in.unsqueeze(2) 
        v_rho = F.softplus(t_c[:, 2:3]).unsqueeze(2) + epsilon
        v_n   = F.softplus(t_c[:, 3:4]).unsqueeze(2) + epsilon
        v_p   = F.softplus(t_c[:, 4:5]).unsqueeze(2) + epsilon

        d_col = torch.sum((p_f - c_f)**2, dim=1, keepdim=True) / (v_c**2 + epsilon)
        d_geo = (torch.sum((p_rho - c_rho)**2, dim=1, keepdim=True) / (v_rho**2 + epsilon) +
                 torch.sum((p_n - c_n)**2, dim=1, keepdim=True) / (v_n**2 + epsilon) +
                 torch.sum((p_p - c_p)**2, dim=1, keepdim=True) / (v_p**2 + epsilon))
        
        return torch.exp(-(d_col + d_geo))

    def compute_eq8(f_in, f_prev_in, weights, theta_alpha_full):
        p_f = get_patches(f_in)
        sum_w = torch.sum(weights, dim=2)
        spatial = torch.sum(p_f * weights, dim=2) / (sum_w + epsilon)
        
        if f_prev_in is not None:
            prev_c = center_crop(f_prev_in)
        alpha = torch.sigmoid(center_crop(theta_alpha_full))
        
        if f_prev_in is None:
            return spatial, sum_w
        return alpha * spatial + (1.0 - alpha) * prev_c, sum_w

    def compute_eq10(f_Ac, f_Bc, sum_wA, sum_wB):
        num = f_Ac * sum_wA + f_Bc * sum_wB
        den = sum_wA + sum_wB
        return num / (den + epsilon)
    
    # Initilization of output buffers
    final_output = torch.zeros((B, 3, H, W), device=device)
    final_fAc    = torch.zeros((B, 3, H, W), device=device)
    final_fBc    = torch.zeros((B, 3, H, W), device=device)

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            y_end, x_end = min(H, y + tile_size), min(W, x + tile_size)
            h_out, w_out = y_end - y, x_end - x
            
            y0, y1 = max(0, y - pad), min(H, y_end + pad)
            x0, x1 = max(0, x - pad), min(W, x_end + pad)
            
            pad_top = pad - (y - y0)
            pad_left = pad - (x - x0)
            pad_bottom = (h_out + 2*pad) - (y1 - y0) - pad_top
            pad_right = (w_out + 2*pad) - (x1 - x0) - pad_left
            
            def get_padded_crop(t):
                crop = t[:, :, y0:y1, x0:x1]
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
                return crop

            # Extraction
            t_theta = get_padded_crop(theta)
            t_fA    = get_padded_crop(f_tilde_A)
            t_fB    = get_padded_crop(f_tilde_B)
            if f_prev is not None:
                t_prev  = get_padded_crop(f_prev)
            else:
                t_prev = None
            t_rho   = get_padded_crop(textures)
            t_n     = get_padded_crop(normals)
            t_p     = get_padded_crop(positions)
            
            assert t_fA.shape[2] == h_out + 2*pad and t_fA.shape[3] == w_out + 2*pad, "Padding error"
            
            # Buffer A
            var_A = F.softplus(center_crop(t_theta)[:, 0:1]) + epsilon
            w_A = compute_weights(t_fA, var_A, t_n, t_rho, t_p, t_theta)
            f_Ac_log, sw_A = compute_eq8(t_fA, t_prev, w_A, t_theta[:, 5:6])
            
            # Buffer B (Log domain)
            var_B = F.softplus(center_crop(t_theta)[:, 1:2]) + epsilon
            w_B = compute_weights(t_fB, var_B, t_n, t_rho, t_p, t_theta)
            f_Bc_log, sw_B = compute_eq8(t_fB, t_prev, w_B, t_theta[:, 5:6])
            
            # Fusion (Log domain)
            f_final_log = compute_eq10(f_Ac_log, f_Bc_log, sw_A, sw_B)
            
            def process_output_tile(tile_log):
                tile_lin = torch.expm1(tile_log)       # Exp
                tile_lin = torch.clamp(tile_lin, min=0.0) # Clamp
                if torch.isnan(tile_lin).any():        # NaN check
                    tile_lin = torch.nan_to_num(tile_lin, nan=0.0)
                return tile_lin
            
            # Final Image
            final_output[:, :, y:y_end, x:x_end] = process_output_tile(f_final_log)
            
            # Buffer A
            final_fAc[:, :, y:y_end, x:x_end] = process_output_tile(f_Ac_log)
            
            # Buffer B
            final_fBc[:, :, y:y_end, x:x_end] = process_output_tile(f_Bc_log)
            
            del w_A, w_B, f_Ac_log, f_Bc_log, f_final_log, t_theta

    return final_fAc, final_fBc, final_output
