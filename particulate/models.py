from typing import List, Optional, Tuple
import math

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
import torch
import torch.nn as nn

from particulate.inference_utils import extract_motion_hierarchy
from particulate.matcher import HungarianMatcher
from particulate.articulation_utils import closest_point_on_axis_to_revolute_plucker


class PositionalEmbedder(nn.Module):
    def __init__(
        self,
        frequency_embedding_size: int,
        hidden_size: int,
        input_dim: int,
        raw: bool = False,
    ):
        super(PositionalEmbedder, self).__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.raw = raw
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size * input_dim + (input_dim if raw else 0), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_dim = input_dim

    @staticmethod
    def pos_embedding(x, dim, max_period=10000, mult_factor: float = 1000.0):
        x = mult_factor * x
        half = dim // 2
        # freqs = torch.exp(torch.arange(half, dtype=torch.float32) / half).to(x.device)
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            x.device
        )
        args = x[..., None].float() * freqs[None, :]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[..., :1])], dim=-1)
        return embeddings

    def forward(
        self, 
        x: torch.FloatTensor
    ):
        assert x.shape[-1] == self.input_dim
        x_embed = self.pos_embedding(x, self.frequency_embedding_size)
        x_embed = x_embed.flatten(start_dim=-2)  # Flatten first: (..., input_dim * frequency_embedding_size)
        if self.raw:
            x_embed = torch.cat([x_embed, x], dim=-1)  # Now concatenate: (..., input_dim * frequency_embedding_size + input_dim)
        x_embed = self.mlp(x_embed)
        return x_embed


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float,
    ):
        super(Block, self).__init__()

        # Query self-attention
        self.norm1 = FP32LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn1 = Attention(
            query_dim=hidden_size,
            dim_head=hidden_size // n_heads,
            heads=n_heads,
            dropout=dropout,
            qk_norm="rms_norm",
            eps=1e-6,
            bias=False
        )

        # Point cloud to query cross-attention
        self.norm2 = FP32LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn2 = Attention(
            query_dim=hidden_size,
            dim_head=hidden_size // n_heads,
            heads=n_heads,
            dropout=dropout,
            qk_norm="rms_norm",
            eps=1e-6,
            bias=False
        )

        # Query to point cloud cross-attention
        self.norm3 = FP32LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn3 = Attention(
            query_dim=hidden_size,
            dim_head=hidden_size // n_heads,
            heads=n_heads,
            dropout=dropout,
            qk_norm="rms_norm",
            eps=1e-6,
            bias=False
        )

        self.final_norm_x = FP32LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.final_norm_q = FP32LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.mlp_x = FeedForward(dim=hidden_size, dropout=dropout)
        self.mlp_q = FeedForward(dim=hidden_size, dropout=dropout)

    def forward(
        self,
        x: torch.FloatTensor, 
        q: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # 1. query self-attention
        q = self.norm1(q)
        q = self.attn1(q) + q

        # 2. point cloud to query cross-attention
        q = q + self.attn2(self.norm2(q), x)

        # 3. query to point cloud cross-attention
        x = x + self.attn3(self.norm3(x), q)

        # 4. final MLP
        x = x + self.mlp_x(self.final_norm_x(x))
        q = q + self.mlp_q(self.final_norm_q(q))

        return x, q


class PAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        use_normals: bool = False,
        max_parts: int = 128,
        use_part_id_embedding: bool = True,
        use_raw_coords: bool = False,
        use_point_features_for_motion_decoding: bool = False,
        point_feature_random_ratio: float = 0.0,
        num_mask_hypotheses: int = 1,
        motion_representation: str = 'per_part_plucker',  # one of ["per_part_plucker", "per_point_closest"]
    ):
        super(PAT, self).__init__()

        self.feat_proj = nn.Linear(input_dim, hidden_size)
        self.pos_embed = PositionalEmbedder(
            frequency_embedding_size=64, 
            hidden_size=hidden_size, 
            input_dim=3,
            raw=use_raw_coords
        )

        self.use_normals = use_normals
        if use_normals:
            self.normal_embed = PositionalEmbedder(
                frequency_embedding_size=64, 
                hidden_size=hidden_size, 
                input_dim=3,
                raw=use_raw_coords
            )

        self.blocks = nn.ModuleList([
            Block(
                hidden_size=hidden_size,
                n_heads=n_heads,
                dropout=dropout,
            ) 
            for _ in range(num_layers)
        ])

        # Decoders
        self.num_mask_hypotheses = num_mask_hypotheses
        if self.num_mask_hypotheses == 1:
            self.point_mask_decoder = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, 1)
            )
            self.point_mask_decoding_func = self._point_mask_decoding_func_single
        else:
            self.point_mask_decoder = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size * 4),
                    nn.SiLU(),
                    nn.Linear(hidden_size * 4, 1)
                )
                for _ in range(self.num_mask_hypotheses)
            ])
            self.point_mask_decoding_func = self._point_mask_decoding_func_multi

        self.use_point_features_for_motion_decoding = use_point_features_for_motion_decoding
        self.point_feature_random_ratio = point_feature_random_ratio

        part_hierarchy_input_dim = hidden_size * 4 if self.use_point_features_for_motion_decoding else hidden_size * 2
        self.part_hierarchy_decoder = nn.Sequential(
            nn.Linear(part_hierarchy_input_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, 1)
        )
        
        motion_input_dim = hidden_size * 2 if self.use_point_features_for_motion_decoding else hidden_size
        self.part_motion_classifier = nn.Sequential(
            nn.Linear(motion_input_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, 4)  # 4 classes: 0 --> "no motion", 1 --> "revolute", 2 --> "prismatic", 3 --> "both"
        )

        self.motion_representation = motion_representation
        motion_input_dim = hidden_size * 2 if self.use_point_features_for_motion_decoding else hidden_size
        self.revolute_motion_decoder = nn.Sequential(
            nn.Linear(motion_input_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, (6 if self.motion_representation == 'per_part_plucker' else 3) + 2)  # Plucker coordinate in R^6 and low & high limits.
        )
        self.prismatic_motion_decoder = nn.Sequential(
            nn.Linear(motion_input_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, 3 + 2)  # Axis only in R^3 and low & high limits.
        )
        if self.motion_representation == 'per_point_closest':
            motion_input_dim = hidden_size * 3 if self.use_point_features_for_motion_decoding else hidden_size * 2
            self.point_motion_decoder = nn.Sequential(
                nn.Linear(motion_input_dim, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, 3)  # Closest point on the axis to the point
            )

        self.max_parts = max_parts
        self.use_part_id_embedding = use_part_id_embedding
        if use_part_id_embedding:
            self.part_id_embed = PositionalEmbedder(
                frequency_embedding_size=64, 
                hidden_size=hidden_size, 
                input_dim=1, 
                raw=False
            )

        self.matcher = HungarianMatcher()

    def _point_mask_decoding_func_single(self, p, q):
        return [(
            self.point_mask_decoder(torch.cat([
                p.unsqueeze(2).expand(-1, -1, q.size(1), -1),  # (B, N, M, D)
                q.unsqueeze(1).expand(-1, p.size(1), -1, -1)   # (B, N, M, D)
            ], dim=-1)).squeeze(-1)
        )]

    def _point_mask_decoding_func_multi(self, p, q):
        return [(
            self.point_mask_decoder[i](torch.cat([
                p.unsqueeze(2).expand(-1, -1, q.size(1), -1),  # (B, N, M, D)
                q.unsqueeze(1).expand(-1, p.size(1), -1, -1)   # (B, N, M, D)
            ], dim=-1)).squeeze(-1)
        ) for i in range(self.num_mask_hypotheses)]

    def forward_attn(
        self,
        xyz: torch.FloatTensor,
        feats: torch.FloatTensor,
        query_xyz: Optional[torch.FloatTensor] = None,
        query_feats: Optional[torch.FloatTensor] = None,
        normals: Optional[torch.FloatTensor] = None,
        text_prompts: Optional[List[str]] = None,
    ):
        batch_size = xyz.shape[0]
        x = self.feat_proj(feats) + self.pos_embed(xyz)
        if self.use_normals:
            assert normals is not None
            x = x + self.normal_embed(normals)

        if text_prompts is not None:
            raise NotImplementedError("Text prompts are not implemented yet")

        assert query_xyz is not None or query_feats is not None or self.use_part_id_embedding
        q = 0
        if self.use_part_id_embedding:
            num_parts = self.max_parts if query_xyz is None else query_xyz.shape[1]
            part_indices = torch.arange(num_parts, device=x.device, dtype=torch.float32) / num_parts
            q = self.part_id_embed(part_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1))
        if query_xyz is not None:
            q = q + self.pos_embed(query_xyz)
        if query_feats is not None:
            q = q + self.feat_proj(query_feats)

        for block in self.blocks:
            x, q = block(x, q)

        return x, q

    def forward_results(
        self,
        xyz: torch.FloatTensor,
        feats: torch.FloatTensor,
        query_xyz: Optional[torch.FloatTensor] = None,
        query_feats: Optional[torch.FloatTensor] = None,
        normals: Optional[torch.FloatTensor] = None,
        text_prompts: Optional[List[str]] = None,
        forward_motion_class: bool = True,
        forward_motion_params: bool = True,
        gt_part_ids: Optional[torch.LongTensor] = None,
        overwrite_part_ids: Optional[torch.LongTensor] = None,
        num_valid_parts: Optional[torch.LongTensor] = None,
        run_matching: bool = False,
        force_hyp_idx: int = -1,
        min_part_confidence: float = 0.0
    ):
        batch_size, num_points = xyz.shape[:2]
        x, q = self.forward_attn(xyz, feats, query_xyz, query_feats, normals, text_prompts)

        point_masks = self.point_mask_decoding_func(x, q)  # (B, M, N)
        
        best_point_mask_id = None
        if self.num_mask_hypotheses == 1:
            point_mask = point_masks[0]
            best_point_mask_id = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        elif force_hyp_idx >= 0:
            point_mask = point_masks[force_hyp_idx]
        else:
            all_point_masks = torch.cat(point_masks, dim=0)
            all_gt_part_ids = torch.cat([gt_part_ids] * self.num_mask_hypotheses, dim=0)
            all_num_valid_parts = torch.cat([num_valid_parts] * self.num_mask_hypotheses, dim=0)

            if run_matching:
                matching = self.matcher(all_point_masks, all_gt_part_ids, all_num_valid_parts)
                sorted_indices = self.get_sorted_indices(matching)
                all_point_masks = self.sort_result(all_point_masks.permute(0, 2, 1), sorted_indices).permute(0, 2, 1)

            all_point_mask_losses = self.compute_point_mask_loss(
                all_point_masks.detach(), all_gt_part_ids, all_num_valid_parts,
                reduction='none'
            ).reshape(batch_size * self.num_mask_hypotheses, num_points).mean(-1).split(batch_size)
            best_point_mask_id = torch.argmin(torch.stack(all_point_mask_losses, dim=0), dim=0)
            
            stacked_point_masks = torch.stack(point_masks, dim=0)  # (num_hypotheses, batch_size, M, N)
            batch_indices = torch.arange(batch_size, device=best_point_mask_id.device)
            point_mask = stacked_point_masks[best_point_mask_id, batch_indices]
            point_mask = point_mask + stacked_point_masks.sum(dim=0) * 0.0 # ensure all hypotheses receive gradients

        sorted_indices = None
        if run_matching:
            assert num_valid_parts is not None and gt_part_ids is not None
            matching = self.matcher(point_mask, gt_part_ids, num_valid_parts)
            sorted_indices = self.get_sorted_indices(matching)
            q = self.sort_result(q, sorted_indices)
            point_mask = self.sort_result(point_mask.permute(0, 2, 1), sorted_indices).permute(0, 2, 1)

        if self.training:
            assert gt_part_ids is not None
            part_ids = gt_part_ids
        elif overwrite_part_ids is not None:
            part_ids = overwrite_part_ids
        else:
            part_ids = point_mask.argmax(dim=-1)
            # Mask all columns where no point is affiliated to -torch.inf
            num_parts = point_mask.shape[-1]
            for part_id in range(num_parts):
                if not (part_ids == part_id).any():
                    point_mask[..., part_id] = -torch.inf

            done = False
            while not done:
                done = True
                probs = point_mask.softmax(dim=-1)
                idx = part_ids.long().unsqueeze(-1)
                part_probs = probs.gather(dim=-1, index=idx).squeeze(-1)
                for part_id in part_ids.unique():
                    # Compute part confidence as the average probability
                    part_confidence = (part_probs[part_ids == part_id].log().sum() / (part_ids == part_id).sum()).exp()
                    if part_confidence < min_part_confidence:
                        done = False
                        point_mask[..., part_id] = -torch.inf
                        part_ids = torch.argmax(point_mask, dim=-1)

        if self.use_point_features_for_motion_decoding:
            max_parts = point_mask.shape[-1]
            part_id_mask = part_ids.unsqueeze(-1) == torch.arange(max_parts, device=x.device).unsqueeze(0).unsqueeze(0)  # (B, N, M)
            sample_probs = torch.where(
                part_id_mask,
                1.0 - (self.point_feature_random_ratio if self.training else 0),
                self.point_feature_random_ratio if self.training else 0
            )  # (B, N, M)
            point_to_part_mask = torch.bernoulli(sample_probs)  # (B, N, M)
                
            part_features = torch.einsum('bnd,bnm->bmd', x, point_to_part_mask)  # (B, M, D)
            counts = point_to_part_mask.sum(dim=1)  # (B, M)
            nonzero_mask = counts > 0  # (B, M)
            part_features = torch.where(
                nonzero_mask.unsqueeze(-1),
                part_features / counts.unsqueeze(-1),
                part_features
            )

            q = torch.cat([q, part_features], dim=-1)

        # Prepare input for part hierarchy decoder
        part_adjacency_matrix = self.part_hierarchy_decoder(
            torch.cat([
                q.unsqueeze(2).expand(-1, -1, q.size(1), -1),  # (B, N, N, D)
                q.unsqueeze(1).expand(-1, q.size(1), -1, -1)   # (B, N, N, D)
            ], dim=-1) # ( 1, 5, 5)
        ).squeeze(-1)

        (
            part_motion_logits,
            revolute_plucker,
            revolute_range,
            prismatic_axis,
            prismatic_range
        ) = (None, None, None, None, None)

        if forward_motion_class:
            part_motion_logits = self.part_motion_classifier(q)

        closest_point_on_axis = None
        if self.motion_representation == 'per_part_plucker':
            if forward_motion_params and forward_motion_class:
                revolute_motion_params = self.revolute_motion_decoder(q)
                prismatic_motion_params = self.prismatic_motion_decoder(q)
                revolute_plucker, revolute_range = (
                    revolute_motion_params[..., :6], 
                    revolute_motion_params[..., 6:]
                )
                prismatic_axis, prismatic_range = (
                    prismatic_motion_params[..., :3], 
                    prismatic_motion_params[..., 3:]
                )
        elif self.motion_representation == 'per_point_closest':
            if forward_motion_params and forward_motion_class:
                revolute_motion_params = self.revolute_motion_decoder(q)
                prismatic_motion_params = self.prismatic_motion_decoder(q)
                revolute_plucker, revolute_range = (
                    revolute_motion_params[..., :3], 
                    revolute_motion_params[..., 3:]
                )
                prismatic_axis, prismatic_range = (
                    prismatic_motion_params[..., :3], 
                    prismatic_motion_params[..., 3:]
                )

                per_point_q = torch.gather(q, dim=1, index=part_ids.unsqueeze(-1).expand(-1, -1, q.size(-1)))
                motion_decoder_input = torch.cat([x, per_point_q], dim=-1)
                closest_point_on_axis = self.point_motion_decoder(motion_decoder_input)
        return (
            point_mask.contiguous(),
            part_adjacency_matrix,
            part_motion_logits,
            revolute_plucker, revolute_range,
            prismatic_axis, prismatic_range,
            closest_point_on_axis,
            part_ids,
            best_point_mask_id
        )

    def get_sorted_indices(
        self,
        matching: torch.LongTensor
    ):
        """
        Get sorted indices of matching values.
        Columns with valid matches (matching != -1) are placed first, sorted by matching values.
        Columns with invalid matches are placed last.

        Args:
            matching: LongTensor of shape (batch_size, num_columns) with part assignments (-1 for invalid)

        Returns:
            sorted_indices: LongTensor of shape (batch_size, num_columns) with permutation indices
        """
        batch_size, num_columns = matching.shape

        # Create valid mask: (batch_size, num_columns)
        valid_mask = matching > -1
        
        # For sorting, replace -1 with large values so they sort last
        matching_for_sort = matching.clone()
        matching_for_sort[~valid_mask] = num_columns
        
        # Sort columns by matching values within each batch
        sorted_indices = torch.argsort(matching_for_sort, dim=-1)  # (batch_size, num_columns)

        return sorted_indices

    def sort_result(
        self,
        result: torch.Tensor,
        sorted_indices: torch.LongTensor,
    ):
        """
        Reorder columns of a tensor using pre-computed sorted indices.
        
        This method applies a column permutation to the result tensor based on sorted indices
        that were previously computed (e.g., from get_sorted_indices method). The permutation
        reorders columns so that those corresponding to valid part matches appear first.
        
        Args:
            result: Tensor of shape (batch_size, num_columns, ...) to be permuted along the column dimension
            sorted_indices: LongTensor of shape (batch_size, num_columns) containing permutation indices
            
        Returns:
            permuted_result: Tensor with columns reordered according to sorted_indices
        """
        new_dims = [None] * (len(result.shape) - 2)
        expanded_indices = sorted_indices[(..., *new_dims)]
        expanded_indices = expanded_indices.expand(-1, -1, *result.shape[2:])
        permuted_result = torch.gather(result, dim=1, index=expanded_indices)
        return permuted_result

    def compute_point_mask_loss(
        self,
        point_mask_logits: torch.FloatTensor,
        part_ids: torch.LongTensor,
        reduction: str = 'mean'
    ) -> Tuple[dict, Optional[float]]:
        num_parts = point_mask_logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            point_mask_logits.reshape(-1, num_parts), 
            part_ids.reshape(-1),
            reduction=reduction
        )

    def compute_dice_loss(
        self,
        point_mask_logits: torch.FloatTensor,
        part_ids: torch.LongTensor,
        num_valid_parts: torch.LongTensor,
    ) -> Tuple[dict, Optional[float]]:
        """
        Compute soft dice loss for multi-class point segmentation.
        
        Args:
            point_mask_logits: (B, N, M) - predicted point mask logits
            part_ids: (B, N) - ground truth part assignments
            num_valid_parts: (B,) - number of valid parts for each batch
            sorted_indices: (B, M) - sorted indices of part assignments
            smooth: smoothing constant to avoid division by zero
        Returns:
            dice_loss: scalar loss value
        """
        device = point_mask_logits.device
        num_parts = point_mask_logits.shape[-1]

        # Create mask for invalid parts
        invalid_parts_mask = (
            torch.arange(num_parts).unsqueeze(0).unsqueeze(0).to(device) >= \
                num_valid_parts.unsqueeze(-1).unsqueeze(-1)  # (B, 1, M)
        )
        
        point_mask_probs = torch.softmax(point_mask_logits, dim=-1)
        
        # Convert part_ids to one-hot encoding
        part_ids_onehot = torch.zeros_like(point_mask_probs)  # (B, N, M)
        part_ids_onehot.scatter_(-1, part_ids.unsqueeze(-1), 1.0)  # (B, N, M)
        
        # Create mask for valid parts only
        valid_parts_mask = ~invalid_parts_mask  # (B, 1, M)
        
        # Compute dice coefficient for each part separately
        # For each part m: intersection = sum over points where both pred and target are high for part m
        intersection = (point_mask_probs * part_ids_onehot).sum(dim=1)  # (B, M)
        
        # Union = sum of predictions + sum of targets for each part
        pred_sum = point_mask_probs.sum(dim=1)  # (B, M) 
        target_sum = part_ids_onehot.sum(dim=1)  # (B, M)
        
        # Dice coefficient: 2 * intersection / (pred_sum + target_sum)
        dice_scores = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)  # (B, M)
        
        # Only compute loss for valid parts
        valid_parts_per_batch = valid_parts_mask.squeeze(1)  # (B, M)
        dice_loss = 1.0 - dice_scores
        
        # Mask out invalid parts
        dice_loss = dice_loss * valid_parts_per_batch
        
        # Average over valid parts and batches
        total_valid_parts = valid_parts_per_batch.sum()
        if total_valid_parts > 0:
            return dice_loss.sum() / total_valid_parts
        else:
            return self.parameters().__next__().sum() * 0

    def compute_motion_hierarchy_loss(
        self,
        logits_motion_structure: torch.FloatTensor,
        gt_motion_structure: torch.BoolTensor,
        num_valid_parts: torch.LongTensor
    ) -> Tuple[dict, Optional[float]]:
        """
        Compute binary cross-entropy loss for part hierarchy.
        
        Args:
            logits_motion_structure: (B, M, M) - predicted adjacency matrix logits
            gt_motion_structure: (B, M, M) - ground truth adjacency matrix
            num_valid_parts: (B,) - number of valid parts for each batch
            sorted_indices: (B, M) - sorted indices of part assignments
        Returns:
            motion_hierarchy_loss: scalar loss value
        """

        device = logits_motion_structure.device
        max_parts = logits_motion_structure.shape[-1]
        
        # Create 2D mask for valid parts (top-left num_valid_parts × num_valid_parts submatrix)
        row_indices = torch.arange(max_parts, device=device).unsqueeze(0).unsqueeze(-1)  # (1, M, 1)
        col_indices = torch.arange(max_parts, device=device).unsqueeze(0).unsqueeze(0)   # (1, 1, M)
        
        valid_parts_mask = (
            (row_indices < num_valid_parts.unsqueeze(-1).unsqueeze(-1)) &  # (B, M, 1)
            (col_indices < num_valid_parts.unsqueeze(-1).unsqueeze(-1))    # (B, 1, M)
        )  # (B, M, M)
        
        logits_valid = logits_motion_structure[valid_parts_mask]
        gt_valid = gt_motion_structure[valid_parts_mask].float()
        # Compute binary cross-entropy loss only for valid parts
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits_valid,  # (N_valid,)
            gt_valid,  # (N_valid,)
            pos_weight=(gt_valid < 0.5).sum() / (gt_valid > 0.5).sum()
        )

    def compute_part_motion_classification_loss(
        self,
        part_motion_logits: torch.FloatTensor,
        gt_part_motion_class: torch.LongTensor,
        num_valid_parts: torch.LongTensor
    ) -> Tuple[dict, Optional[float]]:
        """
        Compute part motion classification loss.
        
        Args:
            part_motion_logits: (B, M, 4) - predicted motion class logits
            gt_part_motion_class: (B, M) - ground truth motion classes
            num_valid_parts: (B,) - number of valid parts for each batch
            sorted_indices: (B, M) - sorted indices of part assignments
        Returns:
            part_motion_loss: scalar loss value
        """

        # Only compute loss for valid parts
        valid_parts_mask = torch.arange(part_motion_logits.shape[1], device=part_motion_logits.device).unsqueeze(0) < num_valid_parts.unsqueeze(1)  # (B, M)
        
        logits_valid = part_motion_logits[valid_parts_mask]
        gt_valid = gt_part_motion_class[valid_parts_mask]
        
        # Compute cross-entropy loss only for valid parts
        return torch.nn.functional.cross_entropy(
            logits_valid.reshape(-1, 4),  # (B*M, 4)
            gt_valid.reshape(-1),  # (B*M,)
            reduction='mean'
        )

    def compute_motion_axis_losses(
        self,
        revolute_plucker: torch.FloatTensor,
        prismatic_axis: torch.FloatTensor,
        gt_revolute_plucker: torch.FloatTensor,
        gt_prismatic_axis: torch.FloatTensor,
        num_valid_parts: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute motion parameter losses for revolute and prismatic motion.
        
        Args:
            revolute_plucker: (B, M, 6) - predicted revolute motion parameters
            prismatic_axis: (B, M, 3) - predicted prismatic motion parameters
            gt_revolute_plucker: (B, M, 6) - ground truth revolute motion parameters
            gt_prismatic_axis: (B, M, 3) - ground truth prismatic motion parameters
            num_valid_parts: (B,) - number of valid parts for each batch
            sorted_indices: (B, M) - sorted indices of part assignments
        Returns:
            revolute_loss: scalar loss value
            prismatic_loss: scalar loss value
        """

        valid_parts_mask = (
            torch.arange(revolute_plucker.shape[1], device=revolute_plucker.device).unsqueeze(0) 
            < num_valid_parts.unsqueeze(1)
        )

        revolute_loss = self.parameters().__next__().sum() * 0
        # 1. Revolute loss
        valid_revolute_mask = valid_parts_mask & torch.any(gt_revolute_plucker[..., :3] != 0, dim=-1)
        if valid_revolute_mask.any():
            revolute_plucker_valid = revolute_plucker[valid_revolute_mask]
            gt_revolute_plucker_valid = gt_revolute_plucker[valid_revolute_mask]
            revolute_loss = torch.nn.functional.l1_loss(
                revolute_plucker_valid,
                gt_revolute_plucker_valid[..., :revolute_plucker_valid.shape[-1]]
            )

        prismatic_loss = self.parameters().__next__().sum() * 0
        # 2. Prismatic loss
        valid_prismatic_mask = valid_parts_mask & torch.any(gt_prismatic_axis[..., :3] != 0, dim=-1)
        if valid_prismatic_mask.any():
            prismatic_axis_valid = prismatic_axis[valid_prismatic_mask]
            gt_prismatic_axis_valid = gt_prismatic_axis[valid_prismatic_mask]
            prismatic_loss = torch.nn.functional.l1_loss(
                prismatic_axis_valid,
                gt_prismatic_axis_valid
            )
        
        return revolute_loss, prismatic_loss

    def compute_motion_range_losses(
        self,
        revolute_range: torch.FloatTensor,
        prismatic_range: torch.FloatTensor,
        gt_revolute_range: torch.FloatTensor,
        gt_prismatic_range: torch.FloatTensor,
        num_valid_parts: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute motion range losses for revolute and prismatic joints.
        """
        valid_parts_mask = (
            torch.arange(revolute_range.shape[1], device=revolute_range.device).unsqueeze(0) 
            < num_valid_parts.unsqueeze(1)
        )

        valid_revolute_mask = valid_parts_mask & torch.any(gt_revolute_range != 0, dim=-1)
        valid_prismatic_mask = valid_parts_mask & torch.any(gt_prismatic_range != 0, dim=-1)
        
        revolute_range_loss = self.parameters().__next__().sum() * 0
        if valid_revolute_mask.any():
            revolute_range_valid = revolute_range[valid_revolute_mask]
            gt_revolute_range_valid = gt_revolute_range[valid_revolute_mask]
            revolute_range_loss = torch.nn.functional.l1_loss(
                revolute_range_valid, gt_revolute_range_valid
            )

        prismatic_range_loss = self.parameters().__next__().sum() * 0
        if valid_prismatic_mask.any():
            prismatic_range_valid = prismatic_range[valid_prismatic_mask]
            gt_prismatic_range_valid = gt_prismatic_range[valid_prismatic_mask]
            prismatic_range_loss = torch.nn.functional.l1_loss(
                prismatic_range_valid, gt_prismatic_range_valid
            )

        return revolute_range_loss, prismatic_range_loss

    def forward(
        self,
        xyz: torch.FloatTensor,
        feats: torch.FloatTensor,
        part_ids: torch.LongTensor,
        num_valid_parts: torch.LongTensor,
        part_structure_matrix: torch.BoolTensor,
        query_xyz: Optional[torch.FloatTensor] = None,
        query_feats: Optional[torch.FloatTensor] = None,
        gt_part_motion_class: Optional[torch.LongTensor] = None,
        gt_revolute_plucker: Optional[torch.FloatTensor] = None,
        gt_revolute_range: Optional[torch.FloatTensor] = None,
        gt_prismatic_axis: Optional[torch.FloatTensor] = None,
        gt_prismatic_range: Optional[torch.FloatTensor] = None,
        gt_closest_point_on_axis: Optional[torch.FloatTensor] = None,
        normals: Optional[torch.FloatTensor] = None,
        text_prompts: Optional[List[str]] = None,
        run_matching: bool = False,
    ) -> Tuple[dict, torch.LongTensor]:
        """
        Forward pass during training.
        """
        forward_motion_class = gt_part_motion_class is not None
        forward_motion_params = (
            (
                gt_revolute_plucker is not None or \
                gt_revolute_range is not None or \
                gt_prismatic_axis is not None or \
                gt_prismatic_range is not None
            ) and self.motion_representation == 'per_part_plucker'
        ) or (
            (
                gt_revolute_plucker is not None or \
                gt_revolute_range is not None or \
                gt_prismatic_axis is not None or \
                gt_prismatic_range is not None
            ) and gt_closest_point_on_axis is not None and self.motion_representation == 'per_point_closest'
        )
        
        (
            point_mask,
            part_adjacency_matrix,
            part_motion_logits,
            revolute_plucker, revolute_range,
            prismatic_axis, prismatic_range,
            closest_point_on_axis,
            _, best_point_mask_id
        ) = self.forward_results(
            xyz, feats, query_xyz, query_feats,
            normals, text_prompts,
            forward_motion_class, forward_motion_params,
            gt_part_ids=part_ids,
            num_valid_parts=num_valid_parts,
            run_matching=run_matching
        )   

        # Compute losses
        # 1. Point mask loss
        point_mask_loss = self.compute_point_mask_loss(
            point_mask,
            part_ids
        )

        # 2. Dice loss
        dice_loss = self.compute_dice_loss(
            point_mask,
            part_ids,
            num_valid_parts
        )
        
        # 2. Part hierarchy loss
        motion_hierarchy_loss = self.compute_motion_hierarchy_loss(
            part_adjacency_matrix,
            part_structure_matrix,
            num_valid_parts
        )
        
        # 3. Part motion classification loss
        part_motion_classification_loss = self.parameters().__next__().sum() * 0
        if gt_part_motion_class is not None and part_motion_logits is not None and forward_motion_class:
            part_motion_classification_loss = self.compute_part_motion_classification_loss(
                part_motion_logits,
                gt_part_motion_class,
                num_valid_parts
            )
        
        # 4. Motion axis losses (if needed)
        part_motion_axis_loss_revolute = self.parameters().__next__().sum() * 0
        part_motion_axis_loss_prismatic = self.parameters().__next__().sum() * 0
        if forward_motion_params and (
            revolute_plucker is not None and
            prismatic_axis is not None and
            gt_revolute_plucker is not None and
            gt_prismatic_axis is not None
        ):
            part_motion_axis_loss_revolute, part_motion_axis_loss_prismatic = \
                self.compute_motion_axis_losses(
                    revolute_plucker, prismatic_axis,
                    gt_revolute_plucker, gt_prismatic_axis, num_valid_parts
                )

        # 5. Motion range losses (if needed)
        part_motion_range_loss_revolute = self.parameters().__next__().sum() * 0
        part_motion_range_loss_prismatic = self.parameters().__next__().sum() * 0
        if forward_motion_params and (
            revolute_range is not None and
            prismatic_range is not None and
            gt_revolute_range is not None and
            gt_prismatic_range is not None
        ):
            part_motion_range_loss_revolute, part_motion_range_loss_prismatic = \
                self.compute_motion_range_losses(
                    revolute_range, prismatic_range,
                    gt_revolute_range, gt_prismatic_range, num_valid_parts
                )

        # 6. Per point closest point on axis loss (if needed)
        point_closest_point_on_axis_loss = self.parameters().__next__().sum() * 0
        if forward_motion_params and (
            closest_point_on_axis is not None and
            gt_closest_point_on_axis is not None
        ):
            per_point_motion_type = torch.gather(gt_part_motion_class, dim=1, index=part_ids)
            revolute_points_flag = per_point_motion_type.eq(1) | per_point_motion_type.eq(3)
            point_closest_point_on_axis_loss = torch.nn.functional.l1_loss(
                closest_point_on_axis[revolute_points_flag],
                gt_closest_point_on_axis[revolute_points_flag]
            )
                    
        # Combine all losses
        losses = dict(
            point_mask_loss=point_mask_loss,
            dice_loss=dice_loss,
            motion_hierarchy_loss=motion_hierarchy_loss,
            part_motion_classification_loss=part_motion_classification_loss,
            part_motion_axis_loss_revolute=part_motion_axis_loss_revolute,
            part_motion_axis_loss_prismatic=part_motion_axis_loss_prismatic,
            part_motion_range_loss_revolute=part_motion_range_loss_revolute,
            part_motion_range_loss_prismatic=part_motion_range_loss_prismatic,
            point_closest_point_on_axis_loss=point_closest_point_on_axis_loss
        )
        return losses, best_point_mask_id

    def _postprocess_results(
        self,
        point_mask,
        part_adjacency_matrix,
        part_motion_logits,
        revolute_plucker, revolute_range,
        prismatic_axis, prismatic_range,
        closest_point_on_axis,
        part_ids,
    ):
        motion_hierarchy_batch = []
        for batch_idx, single_part_adjacency_matrix in enumerate(part_adjacency_matrix): # (B, N, N)
            unique_part_ids = torch.unique(part_ids[batch_idx])
            
            # Extract submatrix for only the unique part IDs that exist
            submatrix = single_part_adjacency_matrix[unique_part_ids][:, unique_part_ids]
            
            # Extract motion hierarchy from the submatrix
            hierarchy_compressed = extract_motion_hierarchy(submatrix)
            
            # Map back to original indices
            hierarchy_original = []
            for parent_idx, child_idx in hierarchy_compressed:
                original_parent = unique_part_ids[parent_idx].item()
                original_child = unique_part_ids[child_idx].item()
                hierarchy_original.append((original_parent, original_child))
            
            motion_hierarchy_batch.append(hierarchy_original)

        part_motion_class = torch.argmax(part_motion_logits, dim=-1)
        is_part_revolute = part_motion_class.eq(1) | part_motion_class.eq(3)
        is_part_prismatic = part_motion_class.eq(2) | part_motion_class.eq(3)

        # Make sure the plucker and axis parameters are valid
        if revolute_plucker is not None:
            revolute_plucker[..., :3] = revolute_plucker[..., :3] / torch.norm(revolute_plucker[..., :3], dim=-1, keepdim=True).clamp_min(1e-8)
        if prismatic_axis is not None:
            prismatic_axis[..., :3] = prismatic_axis[..., :3] / torch.norm(prismatic_axis[..., :3], dim=-1, keepdim=True).clamp_min(1e-8)

        # Assert that all part IDs in motion hierarchy have at least one associated point
        for batch_idx, hierarchy in enumerate(motion_hierarchy_batch):
            if len(hierarchy) > 0:
                # Get all part IDs mentioned in the hierarchy
                hierarchy_part_ids = set()
                for parent_id, child_id in hierarchy:
                    hierarchy_part_ids.add(parent_id)
                    hierarchy_part_ids.add(child_id)
                
                # Get unique part IDs that actually exist in the point cloud
                existing_part_ids = set(part_ids[batch_idx].cpu().numpy().tolist())
                
                # Assert that all hierarchy part IDs exist in the point cloud
                missing_part_ids = hierarchy_part_ids - existing_part_ids
                assert len(missing_part_ids) == 0, f"Batch {batch_idx}: Part IDs {missing_part_ids} in motion hierarchy have no associated points"

        return dict(
            part_ids=part_ids.cpu().numpy().squeeze(0),
            motion_hierarchy=motion_hierarchy_batch[0],
            is_part_revolute=is_part_revolute.cpu().numpy().squeeze(0),
            is_part_prismatic=is_part_prismatic.cpu().numpy().squeeze(0),
            revolute_plucker=revolute_plucker.cpu().numpy().squeeze(0) if revolute_plucker is not None else None,
            revolute_range=revolute_range.cpu().numpy().squeeze(0) if revolute_range is not None else None,
            prismatic_axis=prismatic_axis.cpu().numpy().squeeze(0) if prismatic_axis is not None else None,
            prismatic_range=prismatic_range.cpu().numpy().squeeze(0) if prismatic_range is not None else None,
            closest_point_on_axis=closest_point_on_axis.cpu().numpy().squeeze(0) if closest_point_on_axis is not None else None,
        )

    @torch.no_grad()
    def infer(
        self,
        xyz: torch.FloatTensor,
        feats: torch.FloatTensor,
        query_xyz: Optional[torch.FloatTensor] = None,
        query_feats: Optional[torch.FloatTensor] = None,
        normals: Optional[torch.FloatTensor] = None,
        text_prompts: Optional[List[str]] = None,
        forward_motion_class: bool = True,
        forward_motion_params: bool = True,
        run_matching: bool = False,
        gt_part_ids: Optional[torch.LongTensor] = None,
        overwrite_part_ids: Optional[torch.LongTensor] = None,
        output_all_hyps: bool = False,
        min_part_confidence: float = 0.0
    ):
        assert xyz.shape[0] == 1, "Only batch size 1 is supported"

        num_valid_parts = None
        if gt_part_ids is not None:
            num_valid_parts = gt_part_ids.max(dim=-1).values + 1

        results = []
        if output_all_hyps:
            for hyp_idx in range(self.num_mask_hypotheses):
                results.append(self.forward_results(
                    xyz, feats, query_xyz, query_feats,
                    normals, text_prompts,
                    forward_motion_class, forward_motion_params,
                    gt_part_ids=gt_part_ids,
                    overwrite_part_ids=overwrite_part_ids,
                    num_valid_parts=num_valid_parts,
                    run_matching=run_matching,
                    force_hyp_idx=hyp_idx,
                    min_part_confidence=min_part_confidence
                ))
        else:
            results.append(self.forward_results(
                xyz, feats, query_xyz, query_feats,
                normals, text_prompts,
                forward_motion_class, forward_motion_params,
                gt_part_ids=gt_part_ids,
                overwrite_part_ids=overwrite_part_ids,
                num_valid_parts=num_valid_parts,
                run_matching=run_matching,
                force_hyp_idx=-1,
                min_part_confidence=min_part_confidence
            ))

        postprocessed_results = [self._postprocess_results(*result[:-1]) for result in results]  # Ignore the last element (best_point_mask_id)
        for postprocessed_result in postprocessed_results:
            if self.motion_representation == 'per_point_closest':
                postprocessed_result['revolute_plucker'] = closest_point_on_axis_to_revolute_plucker(
                    postprocessed_result['closest_point_on_axis'],
                    postprocessed_result['part_ids'],
                    postprocessed_result['is_part_revolute'],
                    postprocessed_result['is_part_prismatic'],
                    postprocessed_result['revolute_plucker']
                )
                
        return postprocessed_results


def PAT_S(**kwargs):
    return PAT(num_layers=6, hidden_size=384, n_heads=6, **kwargs)

def PAT_B(**kwargs):
    return PAT(num_layers=6, hidden_size=768, n_heads=12, **kwargs)

def PAT_L(**kwargs):
    return PAT(num_layers=12, hidden_size=1024, n_heads=16, **kwargs)

def PAT_XL(**kwargs):
    return PAT(num_layers=14, hidden_size=1152, n_heads=16, **kwargs)
