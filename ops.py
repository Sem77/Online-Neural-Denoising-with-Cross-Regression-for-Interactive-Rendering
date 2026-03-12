import torch
import torch.nn.functional as F


def estimate_sigma(img, device="CUDA", epsilon=1e-4):
    kernel = torch.ones((3, 1, 3, 3), device=device) / 8.0
    kernel[:, :, 1, 1] = 0
    mean = F.conv2d(img, kernel, padding=1, groups=3)
    var = torch.sum((img - mean)**2, dim=1, keepdim=True)
    return torch.sqrt(torch.clamp(var, min=epsilon))


@torch.no_grad()
def compute_alpha_beta(y_A, y_B, textures, normals, stride=4, window_size=17):
    """
    Implementation of cross-regression 
    
    Arguments:
        y_A, y_B: Tenseurs (B, 3, H, W) - Noise image
        textures: Tenseur (B, 3, H, W) - Albedo/Texture
        normals: Tenseur (B, 3, H, W) - Normals
    """
    B, C, H, W = y_A.shape
    device = y_A.device
    epsilon = 1e-4
    pad = window_size // 2

    # Estimation of Local Variance (3x3)

    sigma_A = estimate_sigma(y_A, device=device, epsilon=epsilon) # (B, 1, H, W)
    sigma_B = estimate_sigma(y_B, device=device, epsilon=epsilon)

    def get_patches(tensor, stride, window):
        return F.unfold(tensor, kernel_size=window, padding=pad, stride=stride)

    patches_yA = get_patches(y_A, stride, window_size)
    patches_yB = get_patches(y_B, stride, window_size)
    patches_tex = get_patches(textures, stride, window_size)
    patches_norm = get_patches(normals, stride, window_size)
    patches_sigA = get_patches(sigma_A, stride, window_size)
    patches_sigB = get_patches(sigma_B, stride, window_size)

    N = window_size * window_size
    L = patches_yA.shape[-1]
    
    def reshape_to_blocks(p, channels):
        return p.view(B, channels, N, L).permute(0, 3, 2, 1)

    yA_blocks = reshape_to_blocks(patches_yA, 3)
    yB_blocks = reshape_to_blocks(patches_yB, 3)
    tex_blocks = reshape_to_blocks(patches_tex, 3)
    norm_blocks = reshape_to_blocks(patches_norm, 3)
    sigA_blocks = reshape_to_blocks(patches_sigA, 1)
    sigB_blocks = reshape_to_blocks(patches_sigB, 1)

    # Identification of central pixel of each window
    idx_c = N // 2
    yA_c = yA_blocks[:, :, idx_c:idx_c+1, :]
    yB_c = yB_blocks[:, :, idx_c:idx_c+1, :]
    tex_c = tex_blocks[:, :, idx_c:idx_c+1, :]
    norm_c = norm_blocks[:, :, idx_c:idx_c+1, :]
    sigA_c = sigA_blocks[:, :, idx_c:idx_c+1, :]
    sigB_c = sigB_blocks[:, :, idx_c:idx_c+1, :]

    # Computation of relatives caracteristics (xi - xc)
    # Caracteristics A to predict B
    diff_yA = (yA_blocks - yA_c) / (sigA_blocks + sigA_c + epsilon)
    diff_tex = tex_blocks - tex_c
    diff_norm = norm_blocks - norm_c
    
    # X_A (B, L, N, 10) : Intercept (1) + ColorA (3) + Tex (3) + Norm (3)
    X_A = torch.cat([torch.ones((B, L, N, 1), device=device), diff_yA, diff_tex, diff_norm], dim=-1)
    
    # Caracteristics B to predict A
    diff_yB = (yB_blocks - yB_c) / (sigB_blocks + sigB_c + epsilon)
    X_B = torch.cat([torch.ones((B, L, N, 1), device=device), diff_yB, diff_tex, diff_norm], dim=-1)

    # Weights computation
    def compute_weights(y, sig, y_c, sig_c):
        dist_sq = torch.sum((y - y_c)**2, dim=-1, keepdim=True)
        denom = sig**2 + sig_c**2 + epsilon
        return torch.exp(-dist_sq / denom)

    W_A = compute_weights(yA_blocks, sigA_blocks, yA_c, sigA_c) # (B, L, N, 1)
    W_B = compute_weights(yB_blocks, sigB_blocks, yB_c, sigB_c)

    # Solve normal equation  (Xt W X) beta = Xt W Y 
    def solve_cross(X, W, Y_target):
        XtW = X.transpose(-2, -1) * W.transpose(-2, -1)
        
        A_mat = XtW @ X # (B, L, 10, 10)
        B_vec = XtW @ Y_target # (B, L, 10, 3)
        
        reg = 1e-4 * torch.eye(A_mat.size(-1), device=A_mat.device)
        A_mat = A_mat + reg 
        
        try:
            coeffs = torch.linalg.solve(A_mat, B_vec) 
        except torch._C._LinAlgError:
            coeffs = torch.linalg.lstsq(A_mat, B_vec).solution

        return coeffs[:, :, 0, :], coeffs[:, :, 1:, :] # alpha (B, L, 3), beta (B, L, 9, 3)

    alpha_A, beta_A = solve_cross(X_A, W_A, yB_blocks)
    alpha_B, beta_B = solve_cross(X_B, W_B, yA_blocks)

    return alpha_A, beta_A, alpha_B, beta_B, sigma_A, sigma_B


def compute_f_tilde(alpha, beta, y_img, textures, normals, sigma, stride=4, window_size=17):
    """
    Compute image f_tilde from alpha beta parameters
    
    alpha: (B, L, 3) - Optimal Intercepts optimaux
    beta: (B, L, 9, 3) - Optimal Gradients (Coulor, Texture, Normal)
    y_img: (B, 3, H, W) - Noised Input image
    textures, normals: (B, 3, H, W) - G-buffers
    sigma: (B, 1, H, W) - Standard deviation
    """
    B, C, H, W = y_img.shape
    device = y_img.device
    pad = window_size // 2
    N = window_size * window_size # Number of pixels in neighbouring (289)
    L = alpha.shape[1]            # Nomber of centers (1/16 of the image)
    epsilon = 1e-4

    # Extraction of neighbors around centers c
    def get_patches(tensor, ch):
        # (B, ch, H, W) -> (B, L, N, ch)
        p = F.unfold(tensor, kernel_size=window_size, padding=pad, stride=stride)
        return p.view(B, ch, N, L).permute(0, 3, 2, 1)

    y_p = get_patches(y_img, 3)
    tex_p = get_patches(textures, 3)
    norm_p = get_patches(normals, 3)
    sig_p = get_patches(sigma, 1)

    # Identification of the values at the centers xc
    idx_c = N // 2
    y_c = y_p[:, :, idx_c:idx_c+1, :]
    tex_c = tex_p[:, :, idx_c:idx_c+1, :]
    norm_c = norm_p[:, :, idx_c:idx_c+1, :]
    sig_c = sig_p[:, :, idx_c:idx_c+1, :]

    # Compute vector (xi - xc)
    diff_color = (y_p - y_c) / (sig_p + sig_c + epsilon)
    diff_tex = tex_p - tex_c
    diff_norm = norm_p - norm_c
    diff_all = torch.cat([diff_color, diff_tex, diff_norm], dim=-1) # (B, L, N, 9)

    # Local prediction : alpha_c + beta_c^T * (xi - xc)
    # alpha: (B, L, 3), beta: (B, L, 9, 3)
    preds = alpha.unsqueeze(2) + torch.matmul(diff_all, beta) # (B, L, N, 3)

    # Compute weights w_ci
    dist_sq = torch.sum((y_p - y_c)**2, dim=-1, keepdim=True)
    w = torch.exp(-dist_sq / (sig_p**2 + sig_c**2 + epsilon)) # (B, L, N, 1)

    # Final Agregation
    weighted_preds = (preds * w).permute(0, 3, 2, 1).reshape(B, 3 * N, L)
    w_sum_p = w.expand(-1, -1, -1, 3).permute(0, 3, 2, 1).reshape(B, 3 * N, L)

    sum_img = F.fold(weighted_preds, (H, W), window_size, padding=pad, stride=stride)
    norm_img = F.fold(w_sum_p, (H, W), window_size, padding=pad, stride=stride)

    return sum_img / (norm_img + epsilon)
