import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, point_mask, part_ids, num_valid_parts):
        """
        Perform Hungarian matching between predicted columns and ground truth parts.
        
        Args:
            point_mask: (B, N, M) - predicted logits before softmax
            part_ids: (B, N) - ground truth part-id for each point
            num_valid_parts: (B,) - number of available parts in each batch
            
        Returns:
            matched_part_ids: (B, M) - mapped part-id for each column (-1 if unmapped)
        """
        batch_size, num_points, num_columns = point_mask.shape
        device = point_mask.device
        
        # Convert logits to probabilities and log probabilities
        probs = torch.softmax(point_mask, dim=-1)  # (B, N, M)
        log_probs = torch.log_softmax(point_mask, dim=-1)  # (B, N, M)
        
        result = []
        
        for i in range(batch_size):
            n_valid = num_valid_parts[i].item()
            
            if n_valid == 0:
                # No valid parts, all columns get -1
                result.append(torch.full((num_columns,), -1, dtype=torch.long, device=device))
                continue
                
            # Create part masks: (n_valid, N) - True if point belongs to part
            part_masks = (part_ids[i].unsqueeze(0) == torch.arange(n_valid, device=device).unsqueeze(1))  # (n_valid, N)
            
            # Create prediction masks: (N, num_columns) - True if predicted to belong to column
            pred_assignments = torch.argmax(probs[i], dim=-1)  # (N,)
            pred_masks = torch.zeros_like(probs[i], dtype=torch.bool)  # (N, num_columns)
            pred_masks[torch.arange(pred_masks.size(0)), pred_assignments] = True
            
            # Matrix multiplication to compute sum of log probs for each part-column pair
            log_prob_sums = part_masks.float() @ log_probs[i]  # (n_valid, num_columns)
            cost_matrix = -log_prob_sums  # (n_valid, num_columns)
            
            # Apply Hungarian algorithm to find optimal assignment
            # Convert to float32 to avoid BFloat16 compatibility issues with scipy
            row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().float().numpy())
            
            # Create result for this batch
            batch_result = torch.full((num_columns,), -1, dtype=torch.long, device=device)
            batch_result[col_indices] = torch.tensor(row_indices, dtype=torch.long, device=device)
            result.append(batch_result)
        
        return torch.stack(result)
