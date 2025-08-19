'''
Utilities for augmentation. Partly credit to Dr. Jo Schlemper
'''
from os.path import join

import torch
import numpy as np
import torchvision.transforms as deftfx
import dataloaders.image_transforms as myit
import copy
import torch.nn.functional as F

sabs_aug = {
        # turn flipping off as medical data has fixed orientations
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':5,
  'shift':(5,5),
  'shear':5,
  'scale':(0.9, 1.2), 
},
'elastic'   : {'alpha':10,'sigma':5},
'patch': 256,
'reduce_2d': True,
'gamma_range': (0.5, 1.5)
}

sabs_augv3 = {
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':30,
  'shift':(30,30),
  'shear':30,
  'scale':(0.8, 1.3), 
},
'elastic'   : {'alpha':20,'sigma':5}, 
'patch': 256,
'reduce_2d': True,
'gamma_range': (0.2, 1.8)
}

augs = {
    'sabs_aug': sabs_aug,
    'aug_v3': sabs_augv3, # more aggresive
}

def get_geometric_transformer(aug, order=3):
    """
    Fixed version that properly handles RandomAffine and ElasticTransform
    order: interpolation degree. Select order=0 for augmenting segmentation 
    """
    affine     = aug['aug'].get('affine', 0)
    alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
    sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
    # flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

    tfx = []
    # if 'flip' in aug['aug']:
    #     tfx.append(myit.RandomFlip3D(**flip))

    if 'affine' in aug['aug']:
        # Create a wrapper class that handles the M parameter automatically
        class FixedRandomAffine:
            def __init__(self, rotate, shift, shear, scale, scale_iso, order):
                self.random_affine = myit.RandomAffine(
                    rotation_range=rotate,
                    translation_range=shift,
                    shear_range=shear,
                    zoom_range=scale,
                    zoom_keep_aspect=scale_iso,
                    order=order
                )
            
            def __call__(self, image):
                # Generate the transformation matrix automatically
                M = self.random_affine.build_M(image.shape[:2])
                # Apply the transform with the matrix
                return self.random_affine(image, M)
        
        tfx.append(FixedRandomAffine(
            rotate=affine.get('rotate'),
            shift=affine.get('shift'),
            shear=affine.get('shear'),
            scale=affine.get('scale'),
            scale_iso=affine.get('scale_iso', True),
            order=order
        ))

    if 'elastic' in aug['aug']:
        # Create a wrapper for ElasticTransform that only returns the image
        class FixedElasticTransform:
            def __init__(self, alpha, sigma):
                self.elastic = myit.ElasticTransform(alpha, sigma)
            
            def __call__(self, image):
                # ElasticTransform returns (image, dx_params, dy_params)
                # We only want the image for the augutils version
                result = self.elastic(image)
                if isinstance(result, tuple):
                    return result[0]  # Return only the transformed image
                else:
                    return result
        
        tfx.append(FixedElasticTransform(alpha, sigma))
    
    input_transform = deftfx.Compose(tfx)
    return input_transform
# def get_geometric_transformer(aug, order=3):
#     """order: interpolation degree. Select order=0 for augmenting segmentation """
#     affine     = aug['aug'].get('affine', 0)
#     alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
#     sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
#     # flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

#     tfx = []
#     # if 'flip' in aug['aug']:
#     #     tfx.append(myit.RandomFlip3D(**flip))

#     if 'affine' in aug['aug']:
#         tfx.append(myit.RandomAffine(affine.get('rotate'),
#                                      affine.get('shift'),
#                                      affine.get('shear'),
#                                      affine.get('scale'),
#                                      affine.get('scale_iso',True),
#                                      order=order))

#     if 'elastic' in aug['aug']:
#         tfx.append(myit.ElasticTransform(alpha, sigma))
#     input_transform = deftfx.Compose(tfx)
#     return input_transform

def get_intensity_transformer(aug):
    """some basic intensity transforms"""

    def gamma_tansform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img

    return gamma_tansform

def transform_with_label(aug):
    """
    Doing image geometric transform
    Proposed image to have the following configurations
    [H x W x C + CL]
    Where CL is the number of channels for the label. It is NOT in one-hot form
    """

    geometric_tfx = get_geometric_transformer(aug)
    intensity_tfx = get_intensity_transformer(aug)

    def transform(comp, c_label, c_img, use_onehot, nclass, **kwargs):
        """
        Args
        comp:               a numpy array with shape [H x W x C + c_label]
        c_label:            number of channels for a compact label. Note that the current version only supports 1 slice (H x W x 1)
        nc_onehot:          -1 for not using one-hot representation of mask. otherwise, specify number of classes in the label

        """
        comp = copy.deepcopy(comp)
        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        # geometric transform
        _label = comp[..., c_img ]
        _h_label = np.float32(np.arange( nclass ) == (_label[..., None]) )
        comp = np.concatenate( [comp[...,  :c_img ], _h_label], -1 )
        comp = geometric_tfx(comp)
        # round one_hot labels to 0 or 1
        t_label_h = comp[..., c_img : ]
        t_label_h = np.rint(t_label_h)
        assert t_label_h.max() <= 1
        t_img = comp[..., 0 : c_img ]

        # intensity transform
        t_img = intensity_tfx(t_img)

        if use_onehot is True:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis = -1), -1)
        return t_img, t_label

    return transform

def random_crop_support_student(support_images, support_masks, crop_scale=(0.7, 0.9), crop_prob=0.5):
    """
    Apply random crop augmentation to student/target encoder support images and masks.
    
    Args:
        support_images: List of support images [way x shot x [B x C x H x W]]
        support_masks: List of foreground/background masks [way x shot x [B x 1 x H x W]]
        crop_scale: Tuple of (min_scale, max_scale) for crop size as fraction of original
        crop_prob: Probability of applying crop (0.0 to 1.0)
        
    Returns:
        cropped_images: Same structure as input but with random crops applied
        cropped_masks: Corresponding cropped masks
    """
    if np.random.rand() > crop_prob:
        # No cropping, return original
        return support_images, support_masks
    
    cropped_images = []
    cropped_masks = []
    
    for way_imgs, way_masks in zip(support_images, support_masks):
        cropped_way_imgs = []
        cropped_way_masks = []
        
        for shot_imgs, shot_masks in zip(way_imgs, way_masks):
            B, C, H, W = shot_imgs.shape
            
            # Generate random crop parameters (same for all items in this shot)
            scale = np.random.uniform(crop_scale[0], crop_scale[1])
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            
            # Random top-left corner
            top = np.random.randint(0, H - crop_h + 1)
            left = np.random.randint(0, W - crop_w + 1)
            
            # Crop images and resize back to original size
            cropped_shot_imgs = shot_imgs[:, :, top:top+crop_h, left:left+crop_w]
            cropped_shot_imgs = F.interpolate(cropped_shot_imgs, size=(H, W), mode='bilinear', align_corners=False)
            
            # Crop masks and resize back to original size  
            cropped_shot_masks = shot_masks[:, :, top:top+crop_h, left:left+crop_w]
            cropped_shot_masks = F.interpolate(cropped_shot_masks, size=(H, W), mode='bilinear', align_corners=False)
            
            cropped_way_imgs.append(cropped_shot_imgs)
            cropped_way_masks.append(cropped_shot_masks)
            
        cropped_images.append(cropped_way_imgs)
        cropped_masks.append(cropped_way_masks)
    
    return cropped_images, cropped_masks


def random_crop_support_student_v2(support_images, fore_masks, back_masks, crop_scale=(0.85, 0.95), crop_prob=0.3):
    """
    Apply random crop augmentation to student/target encoder support images and both fore/back masks.
    Uses foreground-aware cropping to ensure valid crops.
    
    Args:
        support_images: List of support images [way x shot x [B x C x H x W]]
        fore_masks: List of foreground masks [way x shot x [B x H x W]]
        back_masks: List of background masks [way x shot x [B x H x W]]  
        crop_scale: Tuple of (min_scale, max_scale) for crop size as fraction of original
        crop_prob: Probability of applying crop (0.0 to 1.0)
        
    Returns:
        cropped_images: Same structure as input but with random crops applied
        cropped_fore_masks: Corresponding cropped foreground masks
        cropped_back_masks: Corresponding cropped background masks
    """
    if np.random.rand() > crop_prob:
        # No cropping, return originals
        return support_images, fore_masks, back_masks
    
    cropped_images = []
    cropped_fore_masks = []
    cropped_back_masks = []
    
    for way_idx in range(len(support_images)):
        way_imgs = support_images[way_idx]
        way_fore_masks = fore_masks[way_idx]
        way_back_masks = back_masks[way_idx]
        
        cropped_way_imgs = []
        cropped_way_fore_masks = []
        cropped_way_back_masks = []
        
        for shot_idx in range(len(way_imgs)):
            shot_imgs = way_imgs[shot_idx]  # [B x C x H x W]
            shot_fore_masks = way_fore_masks[shot_idx]  # [B x H x W]  
            shot_back_masks = way_back_masks[shot_idx]  # [B x H x W]
            
            B, C, H, W = shot_imgs.shape
            
            # Use simple center crop with small random offset to preserve foreground
            # This is much safer than bounding box approach
            
            # Generate crop size (very conservative)
            scale = np.random.uniform(crop_scale[0], crop_scale[1])
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            
            # Center crop with small random offset
            center_y, center_x = H // 2, W // 2
            offset_range = min(20, (H - crop_h) // 4, (W - crop_w) // 4)  # Small offset
            
            if offset_range > 0:
                offset_y = np.random.randint(-offset_range, offset_range + 1)
                offset_x = np.random.randint(-offset_range, offset_range + 1)
            else:
                offset_y, offset_x = 0, 0
            
            # Calculate crop coordinates
            top = max(0, min(H - crop_h, center_y - crop_h // 2 + offset_y))
            left = max(0, min(W - crop_w, center_x - crop_w // 2 + offset_x))
            
            # Crop images and resize back to original size
            cropped_shot_imgs = shot_imgs[:, :, top:top+crop_h, left:left+crop_w]
            cropped_shot_imgs = F.interpolate(cropped_shot_imgs, size=(H, W), mode='bilinear', align_corners=False)
            
            # Crop foreground masks and resize back to original size
            shot_fore_masks_4d = shot_fore_masks.unsqueeze(1)  # [B x 1 x H x W]
            cropped_fore_4d = shot_fore_masks_4d[:, :, top:top+crop_h, left:left+crop_w]
            cropped_fore_4d = F.interpolate(cropped_fore_4d, size=(H, W), mode='bilinear', align_corners=False)
            # Apply threshold to restore binary mask after bilinear interpolation
            cropped_shot_fore_masks = (cropped_fore_4d > 0.5).float().squeeze(1)  # [B x H x W]
            
            # Crop background masks and resize back to original size
            shot_back_masks_4d = shot_back_masks.unsqueeze(1)  # [B x 1 x H x W]
            cropped_back_4d = shot_back_masks_4d[:, :, top:top+crop_h, left:left+crop_w]
            cropped_back_4d = F.interpolate(cropped_back_4d, size=(H, W), mode='bilinear', align_corners=False)
            # Apply threshold to restore binary mask after bilinear interpolation
            cropped_shot_back_masks = (cropped_back_4d > 0.5).float().squeeze(1)  # [B x H x W]
            
            cropped_way_imgs.append(cropped_shot_imgs)
            cropped_way_fore_masks.append(cropped_shot_fore_masks)
            cropped_way_back_masks.append(cropped_shot_back_masks)
            
        cropped_images.append(cropped_way_imgs)
        cropped_fore_masks.append(cropped_way_fore_masks)
        cropped_back_masks.append(cropped_way_back_masks)
    
    return cropped_images, cropped_fore_masks, cropped_back_masks

