"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv
from .alpmodule2 import MultiProtoAsWCos
from .contrastive import ContrastiveLoss
from .ssl_attention import SSLAttentionModule, FeatureMaskAttention
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, Encoder
# DEBUG
from util.utils import get_tversky_loss
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv' #gridconv  # using local prototype only. 
# Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, momentum=0.99, temperature=0.7):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.momentum = self.config.get('momentum', 0.99)
        self.temperature = self.config.get('temperature', 0.1)
        self.get_encoder(in_channels)
        self.get_cls()
        # Use two-level contrastive loss for better organ clustering with symmetric negative pairs
        self.contrastive_loss = ContrastiveLoss(
            temperature=self.temperature,
            use_projector=True,
            feature_dim=256  # After localconv dimension
        )
        
        # SSL Attention Modules for online-target feature interaction
        # Feature dimension from ResNet101 encoder is 256 (after localconv)
        self.use_ssl_attention = self.config.get('use_ssl_attention', False)
        self.use_mask_attention = self.config.get('use_mask_attention', False)
        
        # Initialize SSL attention if enabled
        if self.use_ssl_attention:
            self.ssl_attention = SSLAttentionModule(
                feature_dim=256, 
                n_heads=self.config.get('ssl_attention_heads', 4), 
                dropout=self.config.get('ssl_attention_dropout', 0.1), 
                n_layers=self.config.get('ssl_attention_layers', 1)
            )
        
        # Initialize mask attention if enabled
        if self.use_mask_attention:
            self.mask_attention = FeatureMaskAttention(
                feature_dim=256,
                n_heads=self.config.get('ssl_attention_heads', 4),
                dropout=self.config.get('ssl_attention_dropout', 0.1)
            )
        
        # Validation: Only one attention type should be active
        if self.use_ssl_attention and self.use_mask_attention:
            print("⚠️  WARNING: Both SSL and Mask attention are enabled. Using SSL attention only.")
            self.use_mask_attention = False

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        # if self.config['which_model'] == 'dlfcn_res101':
        use_coco_init = self.config['use_coco_init']
        
        self.teacher_encoder = TVDeeplabRes101Encoder(use_coco_init)
        self.student_encoder = TVDeeplabRes101Encoder(use_coco_init)

        # else:
            # raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path)['model'], strict = False)
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsWCos(proto_grid = [proto_hw, proto_hw], 
                                            feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')
    
    # update the student encoder with the teacher encoder using momentum
    def update_student_encoder(self, student_encoder):
        for param_t, param_s in zip(self.teacher_encoder.parameters(), student_encoder.parameters()):
            param_s.data = self.momentum * param_s.data + (1 - self.momentum) * param_t.data    

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        #print(class_ids)
        assert n_ways == 1, "Multi-shot has not been implemented yet" 
        # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        # print(supp_imgs[0][0].shape, qry_imgs[0].shape)- 1 for bothtorch

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        # print(sup_bsize, qry_bsize)

        assert sup_bsize == qry_bsize == 1

        # imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] 
        #                         + [torch.cat(qry_imgs, dim=0),], dim=0)

        

        # img_fts = self.encoder(imgs_concat, low_level = False)
        # fts_size = img_fts.shape[-2:]

        # supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
        #     n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        # qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
        #     n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        # fore_mask = torch.stack([torch.stack(way, dim=0)
        #                          for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        # fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True)
        # back_mask = torch.stack([torch.stack(way, dim=0)
        #                          for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'
        

        imgs_concat_teacher = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        imgs_concat_student = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        # ASSERTION: Ensure the same images are passed to both teacher and student encoders
        assert torch.allclose(imgs_concat_teacher, imgs_concat_student), "Teacher and student received different images!"

        img_fts_teacher = self.teacher_encoder(imgs_concat_teacher, low_level = False)
        img_fts_student = self.student_encoder(imgs_concat_student, low_level = False)

        fts_size = img_fts_teacher.shape[-2:]

        supp_fts_teacher = img_fts_teacher[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        supp_fts_student = img_fts_student[:n_ways * n_shots * sup_bsize].view(n_ways, n_shots, sup_bsize, -1, *fts_size)

        qry_fts_teacher = img_fts_teacher[n_ways * n_shots * sup_bsize:].view(n_queries, qry_bsize, -1, *fts_size)
        qry_fts_student = img_fts_student[n_ways * n_shots * sup_bsize:].view(n_queries, qry_bsize, -1, *fts_size)

        fore_mask_teacher = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        fore_mask_student = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        #fore_mask_teacher = torch.autograd.Variable(fore_mask_teacher, requires_grad = True)
        #fore_mask_student = torch.autograd.Variable(fore_mask_student,rquired_grad=True)
        back_mask_teacher = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
        back_mask_student = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)

        # self.update_student_encoder(self.student_encoder)

        ###### Compute loss ######
        align_loss = 0
        contrastive_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1): # batch dimension, fixed to 1
            fg_masks = [] # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            # re-interpolate support mask to the same size as support feature
            # print(fore_mask.shape, fts_size)
            # fore_mask = fore_mask.squeeze(0)#.squeeze(0)
            # # print(fore_mask.shape)
            # res_fg_msk = F.interpolate(fore_mask, size = fts_size, mode = 'bilinear') #for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
            # fore_mask = fore_mask.unsqueeze(0)#.unsqueeze(0)
            # back_mask = back_mask.squeeze(0)#.squeeze(0)
            # res_bg_msk = F.interpolate(back_mask, size = fts_size, mode = 'bilinear') #for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']
            # back_mask = back_mask.unsqueeze(0)#.unsqueeze(0)
            # res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
            # res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']
            
            # print(fore_mask_teacher.shape)- torch.Size([1, 1, 1, 256, 256])
            # print(fore_mask_student.shape)- ''
            # print(back_mask_teacher.shape)- ''
            # print(back_mask_student.shape)- ''
            # res_fg_msk_teacher = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask_teacher], dim = 0) # [nway, ns, nb, nh', nw']
            # res_bg_msk_teacher = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask_teacher], dim = 0) # [nway, ns, nb, nh', nw']
            # res_fg_msk_student = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask_student], dim = 0) # [nway, ns, nb, nh', nw']
            # res_bg_msk_student = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask_student], dim = 0) # [nway, ns, nb, nh', nw']
            # print(res_fg_msk_teacher.shape)
            # print(res_bg_msk_teacher.shape)
            # print(res_fg_msk_student.shape)
            # print(res_bg_msk_student.shape)
            # print(fts_size)
            # print(supp_fts_teacher.shape)

            # Interpolate masks to feature size for both contrastive loss and classifier
            res_fg_msk_teacher = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') for fore_mask_w in fore_mask_teacher], dim=0)
            res_bg_msk_teacher = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask_teacher], dim=0)
            res_fg_msk_student = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') for fore_mask_w in fore_mask_student], dim=0)
            res_bg_msk_student = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask_student], dim=0)
            # unique_classes = set()
            # for way in range(res_fg_msk_teacher.shape[0]):
            #     mask = res_fg_msk_teacher[way]
            #     unique = torch.unique(mask)
            #     print(f"Way {way} unique values in mask:", unique)
            #     unique_classes.update(unique.cpu().numpy().tolist())
            # print("All unique values across ways:", unique_classes)

            

            binary_fg_msk_teacher = (res_fg_msk_teacher > 0.5).float()  # (n_ways, n_shots, sup_bsize, H, W)
            binary_fg_msk_student = (res_fg_msk_student > 0.5).float()  # (n_ways, n_shots, sup_bsize, H, W)
            #unique_binary_classes = set()
            # for way in range(binary_fg_msk_teacher.shape[0]):
            #     mask = binary_fg_msk_teacher[way]
            #     unique = torch.unique(mask)
            #     print(f"Way {way} unique values in binary fg mask:", unique)
            #     unique_binary_classes.update(unique.cpu().numpy().tolist())
            # print("All unique values across ways in binary fg mask:", unique_binary_classes)

            
            # Reshape support features: (n_ways, n_shots, sup_bsize, C, H, W) -> (B, C, H, W)
            supp_fts_teacher_flat = supp_fts_teacher.view(-1, *supp_fts_teacher.shape[-3:])  # (B, C, H, W)
            supp_fts_student_flat = supp_fts_student.view(-1, *supp_fts_student.shape[-3:])  # (B, C, H, W)
            
            # Reshape binary masks: (n_ways, n_shots, sup_bsize, H, W) -> (B, H, W)
            binary_fg_msk_teacher_flat = binary_fg_msk_teacher.view(-1, *binary_fg_msk_teacher.shape[-2:])  # (B, H, W)
            binary_fg_msk_student_flat = binary_fg_msk_student.view(-1, *binary_fg_msk_student.shape[-2:])  # (B, H, W)
            
            # Apply attention before contrastive loss (if enabled)
            if self.use_ssl_attention:
                # Use SSL attention (self + cross attention)
                enhanced_supp_fts_teacher, enhanced_supp_fts_student, attention_weights = self.ssl_attention(
                    supp_fts_teacher_flat,  # online features (teacher, gradient-updated)
                    supp_fts_student_flat   # target features (student, momentum-updated)
                )
                print("🔍 Using SSL Attention (self + cross attention)")
            elif self.use_mask_attention:
                # Use mask-aware attention with foreground masks
                enhanced_supp_fts_teacher, enhanced_supp_fts_student, attention_weights = self.mask_attention(
                    supp_fts_teacher_flat,  # online features (teacher, gradient-updated)  
                    supp_fts_student_flat,  # target features (student, momentum-updated)
                    binary_fg_msk_teacher_flat,  # teacher foreground mask
                    binary_fg_msk_student_flat   # student foreground mask
                )
                print("🎯 Using Mask Attention (foreground-focused)")
            else:
                # Use original features without attention
                enhanced_supp_fts_teacher = supp_fts_teacher_flat
                enhanced_supp_fts_student = supp_fts_student_flat
                attention_weights = None
                print("⚪ Using Simple Model (no attention)")
            
            # Calculate TRUE self-supervised contrastive loss WITHOUT organ class information
            # Use attention-enhanced features for contrastive learning
            contrastive_loss = self.contrastive_loss(
                enhanced_supp_fts_teacher,  # (B, C, H, W) - attention-enhanced teacher features
                enhanced_supp_fts_student,  # (B, C, H, W) - attention-enhanced student features
                binary_fg_msk_teacher_flat,  # (B, H, W) - binary mask (0 or 1)
                binary_fg_msk_student_flat,  # (B, H, W) - binary mask (0 or 1)
                training=self.training  # Use projector only during training
                # NO ORGAN CLASS IDs - this is now true SSL with attention
            )
            
            # Reshape enhanced features back to original format for classifier
            # From (B, C, H, W) back to (n_ways, n_shots, sup_bsize, C, H, W)
            enhanced_supp_fts_teacher_reshaped = enhanced_supp_fts_teacher.view(
                n_ways, n_shots, sup_bsize, *enhanced_supp_fts_teacher.shape[1:]
            )
            enhanced_supp_fts_student_reshaped = enhanced_supp_fts_student.view(
                n_ways, n_shots, sup_bsize, *enhanced_supp_fts_student.shape[1:]
            )

            scores          = []
            assign_maps     = []
            bg_sim_maps     = []
            fg_sim_maps     = []

            
            _raw_score, _, aux_attr = self.cls_unit(qry_fts_teacher, enhanced_supp_fts_teacher_reshaped, res_bg_msk_teacher, mode = BG_PROT_MODE, 
                                                    fg = False,thresh = BG_THRESH, isval = isval, 
                                                    val_wsize = val_wsize, vis_sim = show_viz  )

            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk_teacher):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts_teacher, enhanced_supp_fts_teacher_reshaped , _msk.unsqueeze(0), fg = True, 
                                                        mode = FG_PROT_MODE, # if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', 
                                                        thresh = FG_THRESH, isval = isval, 
                                                        val_wsize = val_wsize, vis_sim = show_viz  ) #if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            # print(pred.shape)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                try:
                    align_loss_epi = self.alignLoss(qry_fts_teacher[:, epi], pred, supp_fts_teacher[:, :, epi],
                                                    fore_mask_teacher[:, :, epi], back_mask_teacher[:, :, epi])
                    align_loss += align_loss_epi
                except:
                    align_loss += 0
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim = 1)
        bg_sim_maps    = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps    = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps, contrastive_loss



    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear') # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear') # 1, n, h ,w
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), fg = False, mode = BG_PROT_MODE, thresh = BG_THRESH )

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), 
                                                    fg = True, mode = FG_PROT_MODE, # if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', 
                                                    thresh = FG_THRESH )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)
                #loss.append( get_tversky_loss(supp_pred.argmax(dim = 1, keepdim = True), supp_label[None, ...], 0.3, 0.7 ,1.0) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))




# Supervised
# """
# ALPNet
# """
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .alpmodule import MultiProtoAsConv
# from .alpmodule2 import MultiProtoAsWCos
# from .contrastive import ContrastiveLoss
# from .backbone.torchvision_backbones import TVDeeplabRes50Encoder, Encoder
# # DEBUG
# from util.utils import get_tversky_loss
# from pdb import set_trace

# import pickle
# import torchvision

# # options for type of prototypes
# FG_PROT_MODE = 'gridconv+' # using both local and global prototype
# BG_PROT_MODE = 'gridconv' #gridconv  # using local prototype only. 
# # Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# # thresholds for deciding class of prototypes
# FG_THRESH = 0.95
# BG_THRESH = 0.95

# class FewShotSeg(nn.Module):
#     """
#     ALPNet
#     Args:
#         in_channels:        Number of input channels
#         cfg:                Model configurations
#     """
#     def __init__(self, in_channels=3, pretrained_path=None, cfg=None, momentum=0.99, temperature=0.7):
#         super(FewShotSeg, self).__init__()
#         self.pretrained_path = pretrained_path
#         self.config = cfg or {'align': False}
#         self.momentum = self.config.get('momentum', 0.99)
#         self.temperature = self.config.get('temperature', 0.1)
#         self.get_encoder(in_channels)
#         self.get_cls()
#         # Use two-level contrastive loss for better organ clustering with symmetric negative pairs
#         self.contrastive_loss = ContrastiveLoss(
#             temperature=self.temperature
#         )

#     def get_encoder(self, in_channels):
#         # if self.config['which_model'] == 'deeplab_res101':
#         # if self.config['which_model'] == 'dlfcn_res101':
#         use_coco_init = self.config['use_coco_init']
#         self.teacher_encoder = TVDeeplabRes50Encoder(use_coco_init)
#         self.student_encoder = TVDeeplabRes50Encoder(use_coco_init)
#         for p in self.student_encoder.parameters():
#             p.requires_grad=False

#         # else:
#             # raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

#         if self.pretrained_path:
#             self.load_state_dict(torch.load(self.pretrained_path)['model'], strict = False)
#             print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

#     def get_cls(self):
#         """
#         Obtain the similarity-based classifier
#         """
#         proto_hw = self.config["proto_grid_size"]
#         feature_hw = self.config["feature_hw"]
#         assert self.config['cls_name'] == 'grid_proto'
#         if self.config['cls_name'] == 'grid_proto':
#             self.cls_unit = MultiProtoAsWCos(proto_grid = [proto_hw, proto_hw], 
#                                             feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
#         else:
#             raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')
    
#     # update the student encoder with the teacher encoder using momentum
#     def update_student_encoder(self, student_encoder):
#         for param_t, param_s in zip(self.teacher_encoder.parameters(), student_encoder.parameters()):
#             param_s.data = self.momentum * param_s.data + (1 - self.momentum) * param_t.data    

#     def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, class_ids, isval, val_wsize, show_viz = False):
#         """
#         Args:
#             supp_imgs: support images
#                 way x shot x [B x 3 x H x W], list of lists of tensors
#             fore_mask: foreground masks for support images
#                 way x shot x [B x H x W], list of lists of tensors
#             back_mask: background masks for support images
#                 way x shot x [B x H x W], list of lists of tensors
#             qry_imgs: query images
#                 N x [B x 3 x H x W], list of tensors
#             show_viz: return the visualization dictionary
#         """
#         # ('Please go through this piece of code carefully')
#         n_ways = len(supp_imgs)
#         n_shots = len(supp_imgs[0])
#         n_queries = len(qry_imgs)
#         #print(class_ids)
#         assert n_ways == 1, "Multi-shot has not been implemented yet" 
#         # NOTE: actual shot in support goes in batch dimension
#         assert n_queries == 1

#         # print(supp_imgs[0][0].shape, qry_imgs[0].shape)- 1 for bothtorch

#         sup_bsize = supp_imgs[0][0].shape[0]
#         img_size = supp_imgs[0][0].shape[-2:]
#         qry_bsize = qry_imgs[0].shape[0]

#         # print(sup_bsize, qry_bsize)

#         assert sup_bsize == qry_bsize == 1

#         # imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] 
#         #                         + [torch.cat(qry_imgs, dim=0),], dim=0)

        

#         # img_fts = self.encoder(imgs_concat, low_level = False)
#         # fts_size = img_fts.shape[-2:]

#         # supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
#         #     n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
#         # qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
#         #     n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
#         # fore_mask = torch.stack([torch.stack(way, dim=0)
#         #                          for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
#         # fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True)
#         # back_mask = torch.stack([torch.stack(way, dim=0)
#         #                          for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'
        

#         imgs_concat_teacher = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
#                                 + [torch.cat(qry_imgs, dim=0),], dim=0)

#         imgs_concat_student = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
#                                 + [torch.cat(qry_imgs, dim=0),], dim=0)

#         # ASSERTION: Ensure the same images are passed to both teacher and student encoders
#         assert torch.allclose(imgs_concat_teacher, imgs_concat_student), "Teacher and student received different images!"

#         img_fts_teacher = self.teacher_encoder(imgs_concat_teacher, low_level = False)
#         img_fts_student = self.student_encoder(imgs_concat_student, low_level = False)

#         fts_size = img_fts_teacher.shape[-2:]

#         supp_fts_teacher = img_fts_teacher[:n_ways * n_shots * sup_bsize].view(
#             n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
#         supp_fts_student = img_fts_student[:n_ways * n_shots * sup_bsize].view(n_ways, n_shots, sup_bsize, -1, *fts_size)

#         qry_fts_teacher = img_fts_teacher[n_ways * n_shots * sup_bsize:].view(n_queries, qry_bsize, -1, *fts_size)
#         qry_fts_student = img_fts_student[n_ways * n_shots * sup_bsize:].view(n_queries, qry_bsize, -1, *fts_size)

#         fore_mask_teacher = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
#         fore_mask_student = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
#         #fore_mask_teacher = torch.autograd.Variable(fore_mask_teacher, requires_grad = True)
#         #fore_mask_student = torch.autograd.Variable(fore_mask_student,rquired_grad=True)
#         back_mask_teacher = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
#         back_mask_student = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)

#         # self.update_student_encoder(self.student_encoder)

#         ###### Compute loss ######
#         align_loss = 0
#         contrastive_loss = 0
#         outputs = []
#         visualizes = [] # the buffer for visualization

#         for epi in range(1): # batch dimension, fixed to 1
#             fg_masks = [] # keep the way part

#             '''
#             for way in range(n_ways):
#                 # note: index of n_ways starts from 0
#                 mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
#                 mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
#                 fg_masks.append( mean_sup_msk )

#                 mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
#             '''
#             # re-interpolate support mask to the same size as support feature
#             # print(fore_mask.shape, fts_size)
#             # fore_mask = fore_mask.squeeze(0)#.squeeze(0)
#             # # print(fore_mask.shape)
#             # res_fg_msk = F.interpolate(fore_mask, size = fts_size, mode = 'bilinear') #for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
#             # fore_mask = fore_mask.unsqueeze(0)#.unsqueeze(0)
#             # back_mask = back_mask.squeeze(0)#.squeeze(0)
#             # res_bg_msk = F.interpolate(back_mask, size = fts_size, mode = 'bilinear') #for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']
#             # back_mask = back_mask.unsqueeze(0)#.unsqueeze(0)
#             # res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
#             # res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']
            
#             # print(fore_mask_teacher.shape)- torch.Size([1, 1, 1, 256, 256])
#             # print(fore_mask_student.shape)- ''
#             # print(back_mask_teacher.shape)- ''
#             # print(back_mask_student.shape)- ''
#             # res_fg_msk_teacher = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask_teacher], dim = 0) # [nway, ns, nb, nh', nw']
#             # res_bg_msk_teacher = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask_teacher], dim = 0) # [nway, ns, nb, nh', nw']
#             # res_fg_msk_student = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask_student], dim = 0) # [nway, ns, nb, nh', nw']
#             # res_bg_msk_student = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask_student], dim = 0) # [nway, ns, nb, nh', nw']
#             # print(res_fg_msk_teacher.shape)
#             # print(res_bg_msk_teacher.shape)
#             # print(res_fg_msk_student.shape)
#             # print(res_bg_msk_student.shape)
#             # print(fts_size)
#             # print(supp_fts_teacher.shape)

#             # Interpolate masks to feature size for both contrastive loss and classifier
#             res_fg_msk_teacher = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') for fore_mask_w in fore_mask_teacher], dim=0)
#             res_bg_msk_teacher = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask_teacher], dim=0)
#             res_fg_msk_student = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') for fore_mask_w in fore_mask_student], dim=0)
#             res_bg_msk_student = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask_student], dim=0)
#             # unique_classes = set()
#             # for way in range(res_fg_msk_teacher.shape[0]):
#             #     mask = res_fg_msk_teacher[way]
#             #     unique = torch.unique(mask)
#             #     print(f"Way {way} unique values in mask:", unique)
#             #     unique_classes.update(unique.cpu().numpy().tolist())
#             # print("All unique values across ways:", unique_classes)

            

#             binary_fg_msk_teacher = (res_fg_msk_teacher > 0.5).float()  # (n_ways, n_shots, sup_bsize, H, W)
#             binary_fg_msk_student = (res_fg_msk_student > 0.5).float()  # (n_ways, n_shots, sup_bsize, H, W)
#             #unique_binary_classes = set()
#             # for way in range(binary_fg_msk_teacher.shape[0]):
#             #     mask = binary_fg_msk_teacher[way]
#             #     unique = torch.unique(mask)
#             #     print(f"Way {way} unique values in binary fg mask:", unique)
#             #     unique_binary_classes.update(unique.cpu().numpy().tolist())
#             # print("All unique values across ways in binary fg mask:", unique_binary_classes)
#             # Get organ class information from the dataloader
#             # The class_ids are available in the dataloader output
#             # For now, we'll use the way index as organ class (since each way represents one organ)
#             # In a real implementation, you would pass the actual class_ids from the dataloader
            
#             # Create organ class tensors for each way
#             # Each way represents one organ class
#             organ_class_teacher = torch.tensor(class_ids, device=res_fg_msk_teacher.device)
#             organ_class_student = torch.tensor(class_ids, device=res_fg_msk_student.device)

#             # ASSERTION: Ensure excluded classes are not present
#             # EXCLUDED_CLASSES = [2, 3]  # <-- Update this list if your excluded classes change
#             # assert all([cls not in EXCLUDED_CLASSES for cls in class_ids]), f"Excluded class in class_ids: {class_ids}"

#             #print(organ_class_teacher, organ_class_student)
#             #print('siuuuu2')
#             # Expand to match the mask dimensions: (n_ways,) -> (n_ways, n_shots, sup_bsize)
#             organ_class_teacher = organ_class_teacher.unsqueeze(1).unsqueeze(2).expand(n_ways, n_shots, sup_bsize)
#             organ_class_student = organ_class_student.unsqueeze(1).unsqueeze(2).expand(n_ways, n_shots, sup_bsize)

            
#             # Reshape support features: (n_ways, n_shots, sup_bsize, C, H, W) -> (B, C, H, W)
#             supp_fts_teacher_flat = supp_fts_teacher.view(-1, *supp_fts_teacher.shape[-3:])  # (B, C, H, W)
#             supp_fts_student_flat = supp_fts_student.view(-1, *supp_fts_student.shape[-3:])  # (B, C, H, W)
            
#             # Reshape binary masks: (n_ways, n_shots, sup_bsize, H, W) -> (B, H, W)
#             binary_fg_msk_teacher_flat = binary_fg_msk_teacher.view(-1, *binary_fg_msk_teacher.shape[-2:])  # (B, H, W)
#             binary_fg_msk_student_flat = binary_fg_msk_student.view(-1, *binary_fg_msk_student.shape[-2:])  # (B, H, W)
            
#             # Reshape organ class tensors: (n_ways, n_shots, sup_bsize) -> (B,)
#             organ_class_teacher_flat = organ_class_teacher.view(-1)  # (B,)
#             organ_class_student_flat = organ_class_student.view(-1)  # (B,)
            
#             # Calculate supervised contrastive loss with organ class information
#             contrastive_loss = self.contrastive_loss(
#                 supp_fts_teacher_flat,  # (B, C, H, W)
#                 supp_fts_student_flat,  # (B, C, H, W)
#                 binary_fg_msk_teacher_flat,  # (B, H, W) - binary mask (0 or 1)
#                 binary_fg_msk_student_flat,  # (B, H, W) - binary mask (0 or 1)
#                 organ_class_teacher_flat,  # (B,) - organ class IDs
#                 organ_class_student_flat   # (B,) - organ class IDs
#             )
            
            

#             scores          = []
#             assign_maps     = []
#             bg_sim_maps     = []
#             fg_sim_maps     = []

            
#             _raw_score, _, aux_attr = self.cls_unit(qry_fts_teacher, supp_fts_teacher, res_bg_msk_teacher, mode = BG_PROT_MODE, 
#                                                     fg = False,thresh = BG_THRESH, isval = isval, 
#                                                     val_wsize = val_wsize, vis_sim = show_viz  )

#             scores.append(_raw_score)
#             assign_maps.append(aux_attr['proto_assign'])
#             if show_viz:
#                 bg_sim_maps.append(aux_attr['raw_local_sims'])

#             for way, _msk in enumerate(res_fg_msk_teacher):
#                 _raw_score, _, aux_attr = self.cls_unit(qry_fts_teacher, supp_fts_teacher , _msk.unsqueeze(0), fg = True, 
#                                                         mode = FG_PROT_MODE, # if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', 
#                                                         thresh = FG_THRESH, isval = isval, 
#                                                         val_wsize = val_wsize, vis_sim = show_viz  ) #if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'

#                 scores.append(_raw_score)
#                 if show_viz:
#                     fg_sim_maps.append(aux_attr['raw_local_sims'])

#             pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
#             # print(pred.shape)
#             outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

#             ###### Prototype alignment loss ######
#             if self.config['align'] and self.training:
#                 try:
#                     align_loss_epi = self.alignLoss(qry_fts_teacher[:, epi], pred, supp_fts_teacher[:, :, epi],
#                                                     fore_mask_teacher[:, :, epi], back_mask_teacher[:, :, epi])
#                     align_loss += align_loss_epi
#                 except:
#                     align_loss += 0
#         output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
#         output = output.view(-1, *output.shape[2:])
#         assign_maps = torch.stack(assign_maps, dim = 1)
#         bg_sim_maps    = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
#         fg_sim_maps    = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

#         return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps, contrastive_loss, supp_fts_teacher, supp_fts_student, res_fg_msk_teacher, res_fg_msk_student


#     # Batch was at the outer loop
#     def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
#         """
#         Compute the loss for the prototype alignment branch

#         Args:
#             qry_fts: embedding features for query images
#                 expect shape: N x C x H' x W'
#             pred: predicted segmentation score
#                 expect shape: N x (1 + Wa) x H x W
#             supp_fts: embedding fatures for support images
#                 expect shape: Wa x Sh x C x H' x W'
#             fore_mask: foreground masks for support images
#                 expect shape: way x shot x H x W
#             back_mask: background masks for support images
#                 expect shape: way x shot x H x W
#         """
#         n_ways, n_shots = len(fore_mask), len(fore_mask[0])

#         # Masks for getting query prototype
#         pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
#         binary_masks = [pred_mask == i for i in range(1 + n_ways)]

#         # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
#         # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
#         skip_ways = []

#         ### added for matching dimensions to the new data format
#         qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

#         ### end of added part

#         loss = []
#         for way in range(n_ways):
#             if way in skip_ways:
#                 continue
#             # Get the query prototypes
#             for shot in range(n_shots):
#                 img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

#                 qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear') # [1 (way), n (shot), h, w]

#                 # background
#                 qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear') # 1, n, h ,w
#                 scores = []

#                 _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), fg = False, mode = BG_PROT_MODE, thresh = BG_THRESH )

#                 scores.append(_raw_score_bg)

#                 _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), 
#                                                     fg = True, mode = FG_PROT_MODE, # if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', 
#                                                     thresh = FG_THRESH )
#                 scores.append(_raw_score_fg)

#                 supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
#                 supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

#                 # Construct the support Ground-Truth segmentation
#                 supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

#                 supp_label[fore_mask[way, shot] == 1] = 1
#                 supp_label[back_mask[way, shot] == 1] = 0
#                 # Compute Loss
#                 loss.append( F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)
#                 #loss.append( get_tversky_loss(supp_pred.argmax(dim = 1, keepdim = True), supp_label[None, ...], 0.3, 0.7 ,1.0) / n_shots / n_ways)

#         return torch.sum( torch.stack(loss))

