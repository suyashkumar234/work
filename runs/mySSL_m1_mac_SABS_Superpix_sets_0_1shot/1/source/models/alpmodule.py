"""
ALPModule
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
# for unit test from spatial_similarity_module import NONLocalBlock2D, LayerNorm

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map

        """
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)  ]
        self.avg_pool_op = nn.AvgPool2d( kernel_size  )

        # self.conv1x1 = nn.Sequential(nn.Conv2d(256, 256, 3), 
        #                              nn.ReLU(inplace = True), 
        #                              nn.BatchNorm2d(256), 
        #                              nn.Conv2d(256, 1, 1)
        #                              )

    def forward(self, qry, sup_x, sup_y, mode, thresh, fg = True, isval = False, val_wsize = None, vis_sim = False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]
            vis_sim:    visualize raw similarities or not
        """

        qry = qry.squeeze(1) # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        # print(qry.shape)
        sup_x = sup_x.squeeze(0).squeeze(1) # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0) # [nshot, 1, h, w]

        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask': # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C

            proto = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]

            vis_dict = {'proto_assign': None} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]

        # no need to merge with gridconv+
        elif mode == 'gridconv': # using local prototypes only

            input_size = qry.shape
            nch = input_size[1]

            sup_nshot = sup_x.shape[0]
            # print(sup_x.shape)

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )

            # print(n_sup_x.shape)

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # way(1),nb, hw, nc
            # print(n_sup_x.shape)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
            # print(n_sup_x.shape)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            # print(sup_y_g.shape)
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            # print(sup_y_g.shape)

            protos = n_sup_x[sup_y_g > thresh, :] # npro, nc
            # print(protos.shape)
            pro_n = safe_norm(protos)
            # print(pro_n.shape)
            qry_n = safe_norm(qry)
            # print(qry_n.shape)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            debug_assign = dists.argmax(dim = 1).float().detach()

            vis_dict = {'proto_assign': debug_assign} # things to visualize

            if vis_sim: # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict


        elif mode == 'gridconv+': # local and global prototypes

            input_size = qry.shape
            nch = input_size[1]
            nb_q = input_size[0]

            sup_size = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )
            # print(n_sup_x.shape)

            sup_nshot = sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
            # print(n_sup_x.shape)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
            # print(n_sup_x.shape)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            # print(sup_y_g.shape)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            # print(sup_y_g.shape)

            protos = n_sup_x[sup_y_g > thresh, :]


            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            pro_n = safe_norm( torch.cat( [protos, glb_proto], dim = 0 ) )

            qry_n = safe_norm(qry)
            # print(pro_n.shape, qry_n.shape)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            raw_local_sims = dists.detach()


            debug_assign = dists.argmax(dim = 1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        else:
            B,C,H,W = qry.shape
            nch = C #, input_size[1]
            nb_q = B #input_size[0]

            sup_size = sup_x.shape[0]
            # print(sup_size)

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )
            # print(n_sup_x.shape)

            sup_nshot = sup_x.shape[0]
            # print(sup_nshot)

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
            # print(n_sup_x.shape)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            # print(sup_y_g.shape)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            # print(sup_y_g.shape)
            # print(n_sup_x.shape, sup_y_g.shape)

            protos = n_sup_x[sup_y_g > thresh, :] #n_pro x n_c
            protos = protos.unsqueeze(0)
            # print(protos.shape)
            if fg:
                glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
                if F.avg_pool2d(sup_y, 4).max() >= 0.95:
                    protos = torch.cat( [protos, glb_proto.unsqueeze(1)], dim = 0 )
                else:
                    protos = glb_proto.unsqueeze(1)
            # else:
            #     glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
                
                
            # pro_n = safe_norm( protos )

            # qry_n = safe_norm(qry)

            qry_n = qry

            qry_n = qry_n.view(B,C,-1).transpose(1,2)
            # print(qry_n.shape)

            # protos = protos.view(B, protos.shape[0], protos.shape[1])
            # print(protos.shape)

            self.temperature = C**0.5

            attn1 = torch.bmm(protos/self.temperature, qry_n.transpose(1, 2)) #.transpose(1,2) #tranpose added after run 109
            attn1 = F.softmax(attn1/0.05, dim = -1)
            # print(attn1.shape)
            # attn1 = torch.multiply(attn1, mask.transpose(1,2))

            # attn2 = torch.bmm(qry_n/self.temperature, protos.transpose(1, 2)) #.transpose(1,2) #tranpose added after run 109
            # attn2 = F.softmax(attn2/0.05, dim = -1)
            # attn2 = torch.multiply(attn2, mask)

            attn = torch.bmm(attn1.transpose(1,2), attn1)
            # print(attn.shape)
            # attn1 = torch.multiply(attn1, mask.transpose(1,2))

            pred_grid = torch.bmm(attn,qry_n) * 20
            # print(pred_grid.shape)
            pred_grid = self.conv1x1(pred_grid.transpose(1,2).view(B,C,H,W)) #.mean(dim = 1).view(B,1,H,W)


            # pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            raw_local_sims = attn.detach()


            debug_assign = attn.argmax(dim = 1).float().view(B,H,W)

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict
