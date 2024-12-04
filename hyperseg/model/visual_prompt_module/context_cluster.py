from abc import ABC, abstractmethod
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

def rand_sample(x, max_len):
    if x.shape[0] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
    return x[rand_idx, :]

def filter_repeat_point(x, new_reso = 27):
    
    new_non_zero_pos = (x * new_reso).int()
    unique_new_non_zero_pos = torch.unique(new_non_zero_pos, return_inverse=False, dim=0)

    return unique_new_non_zero_pos / new_reso



def rand_sample_repeat(x, max_len, filter_repeat = False, new_reso = 27): # make sure sample total max_len points

    if filter_repeat:
        x = filter_repeat_point(x, new_reso=new_reso)

    if x.shape[0] < max_len:
        indices = torch.randint(0, x.shape[0], (max_len - x.shape[0],))
        # pdb.set_trace()
        return torch.cat((x, x[indices]), dim=0) 
    elif x.shape[0] == max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
        return x[rand_idx, :]


def point_sample(input, point_coords, return_dtype, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float().to(input.device), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output








class region_pooling(nn.Module):
    def __init__(self, num_sample_point):
        super().__init__()
        self.num_sample_point = num_sample_point
        self.pooler = nn.AdaptiveAvgPool1d(output_size=1)

    def extract_region_feature(self, region_feature_map, region_masks, original_dtype, return_dtype, gt_region_masks_list):
        assert len(region_feature_map) == len(region_masks)
        all_points = []
        all_points_fea = []
        all_points_img_ids = []
        for img_id, (region_feature_map_i, region_masks_list_i) in enumerate(zip(region_feature_map, region_masks)):
            # region_feature_map_i: [H*W, C]
            if len(region_masks_list_i) != 0:
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]], device=region_masks_list_i[0].device)[None,]
                # [num_sample_point, 2]
                for m in region_masks_list_i:
                    if m.nonzero().shape[0] <=0:
                        if gt_region_masks_list is not None:
                            region_masks_list_i = gt_region_masks_list[img_id]
                        else:
                            print('error')

                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]
                
                cur_non_zero_pos = [rand_sample_repeat((m.nonzero() / ori_image_wh), self.num_sample_point, filter_repeat = True, new_reso = h) for m
                                    in
                                    region_masks_list_i]
                # [num_mask, num_sample_point, 2]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos)



                dup_region_feature_map_i = region_feature_map_i.reshape(h, w, c).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(cur_non_zero_pos.shape[0], 1, 1,
                                                                                        1)
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                region_feature_i = point_sample(dup_region_feature_map_i_ori_type,
                                                cur_non_zero_pos.flip(dims=(2,)).type(original_dtype),
                                                return_dtype,
                                                align_corners=True,
                                                )
                # [num_mask, num_sample_point, C]
                region_feature_i = region_feature_i.transpose(-2, -1)

                cur_img_id = [img_id] * len(cur_non_zero_pos)

                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_id)

        return all_points, all_points_fea, all_points_img_ids
    # feature_map: bs im_token_num(729) C(2560)   region_masks(list): bs * region_num * 384 * 384
    def forward(self, feature_map, region_masks, original_dtype, return_dtype, return_all_point=False, gt_region_masks_list=None):
        assert len(feature_map) == len(region_masks)
        batch_size = len(feature_map)
        all_points, all_points_fea, all_points_img_ids = self.extract_region_feature(feature_map, region_masks,
                                                                                     original_dtype, return_dtype, gt_region_masks_list)

        if len(all_points) == 0:
            return [None] * len(region_masks)

        all_points = torch.cat(all_points, dim=0).to(return_dtype)
        all_points_fea = torch.cat(all_points_fea, dim=0).to(return_dtype) # (bs*region_num) * num_sample_point(256) * 1024
        all_points_img_ids = torch.tensor(all_points_img_ids, device=all_points_fea.device)

        if return_all_point:
            region_feat = all_points_fea
        # average among 256 sample point
        else:
            region_feat = self.pooler(all_points_fea.transpose(-2, -1)).transpose(-2, -1) # (bs*region_num) * 1 * 1024
            

        region_feature_list = []
        for bs in range(batch_size):
            index = all_points_img_ids == bs
            region_feature_list.append(region_feat[index])
        return region_feature_list










