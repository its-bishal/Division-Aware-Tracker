import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

class LDDMMMatcher(nn.Module):
    def __init__(self, iterations=50, step_size=0.01, sigma=10.0, 
                 loss_type='mse', blur=0.05, reach=0.5, device='cpu'):
        """
        Simplified PyTorch-based Large Deformation Diffeomorphic Metric Mapping (LDDMM) inspired matching.
        Instead of full EPDiff PDE solving, this uses a heavily regularized displacement field optimization
        to approximate diffeomorphic flow for shape matching.
        """
        super().__init__()
        self.iterations = iterations
        self.step_size = step_size
        self.sigma = sigma
        self.loss_type = loss_type
        self.device = device
        
        if self.loss_type == 'sinkhorn':
            if SamplesLoss is None:
                raise ImportError("geomloss is required for Sinkhorn loss. Please pip install geomloss.")
            self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, reach=reach)
        
        # Create a smoothing kernel for regularization
        kernel_size = int(sigma * 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        self.kernel = kernel / kernel.sum()
        self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size).to(self.device)
        self.padding = kernel_size // 2

    def smooth_vector_field(self, v):
        """Apply Gaussian smoothing to regularize the vector field, enforcing diffeomorphic-like properties."""
        v_x = F.conv2d(v[:, 0:1, :, :], self.kernel, padding=self.padding)
        v_y = F.conv2d(v[:, 1:2, :, :], self.kernel, padding=self.padding)
        return torch.cat([v_x, v_y], dim=1)

    def forward(self, source_mask, target_mask):
        """
        source_mask: [H, W] numpy array or torch tensor (binary mask of parent)
        target_mask: [H, W] numpy array or torch tensor (binary mask of potential daughters)
        Returns:
            energy: float, total LDDMM-like deformation cost
        """
        if not isinstance(source_mask, torch.Tensor):
            source = torch.tensor(source_mask, dtype=torch.float32, device=self.device)
        else:
            source = source_mask.to(self.device).float()
            
        if not isinstance(target_mask, torch.Tensor):
            target = torch.tensor(target_mask, dtype=torch.float32, device=self.device)
        else:
            target = target_mask.to(self.device).float()
            
        source = source.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        target = target.unsqueeze(0).unsqueeze(0)
        
        H, W = source.shape[-2:]
        
        # Initialize stationary velocity field
        v = torch.zeros((1, 2, H, W), dtype=torch.float32, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([v], lr=self.step_size)
        
        # Create base grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        base_grid = torch.stack([x, y], dim=-1).unsqueeze(0) # [1, H, W, 2]
        
        best_energy = float('inf')
        
        for _ in range(self.iterations):
            optimizer.zero_grad()
            
            # Smooth velocity field
            v_smooth = self.smooth_vector_field(v)
            
            # Integrate displacement (simplified one-step Euler for speed, instead of full scaling and squaring)
            # v_smooth is [1, 2, H, W]. Need to convert to grid [1, H, W, 2]
            displacement = v_smooth.permute(0, 2, 3, 1) 
            
            # Apply displacement to base grid
            grid = base_grid + displacement
            
            # Warp source image
            warped_source = F.grid_sample(source, grid, align_corners=True, padding_mode='zeros')
            
            # Calculate loss (Image similarity + regularization)
            if self.loss_type == 'sinkhorn':
                # Instead of downsampling, extract only the "active" pixels (foreground).
                # This drastically reduces the number of points for Optimal Transport
                # while preserving 100% of the original resolution and accuracy!
                
                active_mask = (warped_source > 1e-3) | (target > 1e-3)
                if not active_mask.any():
                    # Fallback if somehow perfectly empty
                    active_mask[..., H//2, W//2] = True
                    
                active_mask_flat = active_mask.view(-1)
                
                # Extract weights for active pixels only
                weights_source = warped_source.view(-1)[active_mask_flat].unsqueeze(0)
                weights_target = target.view(-1)[active_mask_flat].unsqueeze(0)
                
                # Generate full grid and extract coords for active pixels
                y_d, x_d = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=self.device),
                    torch.linspace(-1, 1, W, device=self.device),
                    indexing='ij'
                )
                full_grid = torch.stack([x_d, y_d], dim=-1).view(-1, 2)
                coords = full_grid[active_mask_flat].unsqueeze(0)
                
                # Sinkhorn divergence on the sparse subset
                sim_loss = self.sinkhorn_loss(weights_source, coords, weights_target, coords)
            else:
                # Default Image similarity: MSE
                sim_loss = F.mse_loss(warped_source, target)
            
            # Regularization: kinetic energy of the velocity field
            reg_loss = torch.mean(v_smooth ** 2)
            
            # Total energy
            # The trade-off parameter lambda is baked into step size and sigma here for simplicity
            loss = sim_loss + 0.1 * reg_loss
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                current_energy = loss.item()
                best_energy = min(current_energy, best_energy)
                    
        return best_energy

def compute_lddmm_cost(parent_mask, daughter_mask, config):
    """
    Helper function to compute cost given configs.
    """
    matcher = LDDMMMatcher(
        iterations=config.lddmm_iterations,
        step_size=config.lddmm_step_size,
        sigma=config.lddmm_sigma,
        loss_type=getattr(config, 'lddmm_loss_type', 'mse'),
        blur=getattr(config, 'lddmm_sinkhorn_blur', 0.05),
        reach=getattr(config, 'lddmm_sinkhorn_reach', 0.5),
        device='cuda'
    )
    
    with torch.enable_grad(): 
        energy = matcher(parent_mask, daughter_mask)
        
    return energy
