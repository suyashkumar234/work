"""
Manually labeled dataset
TODO: 
1. Merge with superpixel dataset
"""
import glob
import numpy as np
import dataloaders.augutils as myaug
import torch
import random
import os
import copy
import platform
import json
import re
from dataloaders.common import BaseDataset, Subset
# from common import BaseDataset, Subset
from dataloaders.dataset_utils import* # importing all public item from datset_utils
from pdb import set_trace
from util.utils import CircularList
# ManualAnnoDataset used by both training and validation cause of supervised setup, traversed 2 times
class ManualAnnoDataset(BaseDataset): # inheriting functions from class BaseDataset
    def __init__(self, 
                which_dataset,
                base_dir,
                idx_split, 
                mode, 
                transforms, 
                scan_per_load,
                min_fg = '', 
                fix_length = None, 
                tile_z_dim = 3, 
                nsup = 1, 
                exclude_list = [], 
                extern_normalize_func = None,**kwargs):
        """
        Manually labeled dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation, like idx_split=0 using fold 0 for validation and training in other folds
            mode:               'train', 'val'. 
            transforms:         data transform (augmentation) function
            min_fg:             minimum number of positive pixels in a 2D slice, mainly for stablize training when trained on manually labeled dataset
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time. How many patient to load into the memory at once
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images.artificially increases the number of channels in medical images to make them compatible with deep learning models designed for RGB images.tile_z_dim=3 takes one grayscale medical slice and duplicates it 3 times to create a fake "RGB" image.
            nsup:               number of support scans
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            extern_normalize_function:  normalization function used for data pre-processing-? 
        """
        super(ManualAnnoDataset, self).__init__(base_dir)
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.fix_length = fix_length
        self.all_label_names = self.label_name
        self.nclass = len(self.label_name) # total number of classes in the dataset
        self.tile_z_dim = tile_z_dim
        self.base_dir = base_dir
        self.nsup = nsup
        # img_pid gives the patient id
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ] # getting patient ids-['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '4', '5', '6', '7', '8', '9']
        #print(self.img_pids) - list of all the patient ids are extracted and will be further splitted for training or validation depending upon the mode and idx_split
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x))) # make it circular for the ease of spliting folds
        #sorted list 
        self.exclude_lbs = exclude_list
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')

        self.idx_split = idx_split
        #print(idx_split)- 0 as taking fold-0
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold , get_scanids function declared later
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg) # checking the instace of the the class, taking min_fg as str
        print(f"Using min_fg = '{self.min_fg}' for classmap file")
        self.scan_per_load = scan_per_load
        

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold-
        #print(self.img_lb_fids)-{'7': {'img_fid': 'E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized\\image_7.nii.gz', 'lbs_fid': 'E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized\\label_7.nii.gz'}

        if extern_normalize_func is not None: # helps to keep consistent between training and testing dataset.
            self.norm_func = extern_normalize_func
            print(f'###### Dataset: using external normalization statistics ######') # for testing or validation 
        else:
            self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])
            print(f'###### Dataset: using normalization statistics calculated from loaded data ######')  # for training

        if self.is_train: # if true 
            if scan_per_load > 0: # buffer needed
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
                #print('a')
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
                #print('b') since scan_per_load is equakl to -1 we are loading all the scan_ids
                #print(self.pid_curr_load)-['7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
            #print(self.pid_curr_load)-['0', '1', '2', '3', '4', '5', '6']- why? check why adding one more support do analysis by changing the fold value
            self.potential_support_sid = []
        else:
            raise Exception
        # print(self.scan_ids, self.pid_curr_load)
        self.actual_dataset = self.read_dataset()
        #print(self.actual_dataset )- prints a dictionary img data as 'img' and data shape is (256,256,1), label data similar to img data and other meta data 'is_start': False, 'is_end': False, 'nframe': -1, 'scan_id': '0', 'z_id': 6
        self.size = len(self.actual_dataset)
        #print(self.size)-total number of slice loaded from all the patient in the current fold 
        self.overall_slice_by_cls = self.read_classfiles()
        #print(self.overall_slice_by_cls) dictionary contatining slice of every organ if we want for training or validation can try self.idx_by_class 
        self.update_subclass_lookup()

    def get_scanids(self, mode, idx_split): # gets the id of patient depending upon training or validation
        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup]) 
        #print(val_ids)-['0', '1', '2', '3', '4', '5', '6']
        self.potential_support_sid = val_ids[-self.nsup:] # this is actual file scan id, not index-['6']. Note- The extra patient added specifically to serve as support provider
        #print(self.potential_support_sid)-['6']
        if mode == 'train':
            #print( [ ii for ii in self.img_pids if ii not in val_ids ])-['7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
            return [ ii for ii in self.img_pids if ii not in val_ids ]
            
        elif mode == 'val':
            return val_ids
        

    def reload_buffer(self): # check
        """
        Reload a portion of the entire dataset, if the dataset is too large
        1. delete original buffer
        2. update self.ids_this_batch
        3. update other internel variables like __len__
        """
        if self.scan_per_load <= 0:
            print("We are not using the reload buffer, doing notiong")
            return -1

        del self.actual_dataset
        del self.info_by_scan
        self.pid_curr_load = np.random.choice( self.scan_ids, size = self.scan_per_load, replace = False )
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.update_subclass_lookup()
        print(f'Loader buffer reloaded with a new size of {self.size} slices')

    def organize_sample_fids(self):
        out_list = {}
        for curr_id in self.scan_ids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            _lb_fid  = os.path.join(self.base_dir, f'label_{curr_id}.nii.gz')

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            out_list[str(curr_id)] = curr_dict
        #getting a dictionary containing the path    
        #print(out_list)-{'7': {'img_fid': 'E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized\\image_7.nii.gz', 'lbs_fid': 'E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized\\label_7.nii.gz'}....... similarly
        return out_list
    
   

    def read_dataset(self):
        """
        Build index pointers to individual slices
        Also keep a look-up table from scan_id, slice to index
        """
        out_list = []
        self.scan_z_idx = {} #scan_z_idx['12'][45] = 1205 means slice 45 from patient 12 is at position 1205
        self.info_by_scan = {} # meta data of each scan(Spatial metadata for each patient's 3D scan)
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset()
        # example-
        # Patient 7, slice 0 → out_list[0] → glb_idx = 0
        # Patient 7, slice 1 → out_list[1] → glb_idx = 1  
        # Patient 7, slice 2 → out_list[2] → glb_idx = 2
        # ...
        # Patient 7, slice 99 → out_list[99] → glb_idx = 99
        # Patient 8, slice 0 → out_list[100] → glb_idx = 100
        # Patient 8, slice 1 → out_list[101] → glb_idx = 101

        for scan_id, itm in self.img_lb_fids.items():
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out

            img = img.transpose(1,2,0) # Reorder axes to (H,W,Z)

            self.info_by_scan[scan_id] = _info

            img = np.float32(img)
            img, mean, std = self.norm_func(img) #Normalize

            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])] # slots, all empty initially([-1,-1,-1......]) are filled with global indeximg 
        #         '7': [0, 1, 2, ..., 99],           # Patient 7: positions 0-99
        #         '8': [100, 101, 102, ..., 219],    # Patient 8: positions 100-219   
        #         '9': [220, 221, 222, ..., 339]     # Patient 9: positions 220-339
        # }


            lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1,2,0)
            lb = np.float32(lb)
            #print(lb.shape)-(257, 257, 149)

            img = img[:256, :256, :] # FIXME a bug in shape from the pre-processing code
            lb = lb[:256, :256, :] #It is slicing the image (img) and label (lb) arrays to ensure they are cropped to a fixed size of 256 x 256, while keeping all channels intact.
            #print(lb.shape)-(256, 256, 149)
            assert img.shape[-1] == lb.shape[-1] # checking that img and lb both have the same number of channels 
            base_idx = img.shape[-1] // 2 # index of the middle slice
            
            # write the beginning frame
            out_list.append( {"img": img[..., 0: 1],
                            "mean":mean,
                            "std":std,
                           "lb":lb[..., 0: 0 + 1],
                           "is_start": True,
                           "is_end": False,
                           "nframe": img.shape[-1],
                           "scan_id": scan_id,
                           "z_id":0})

            self.scan_z_idx[scan_id][0] = glb_idx # ?
            glb_idx += 1
        #   out_list = [
        #     {"img": slice_0_data, "scan_id": '7', "z_id": 0, "is_start": True, ...},   # Position 0
        #     {"img": slice_1_data, "scan_id": '7', "z_id": 1, "is_start": False, ...}, # Position 1
        #     {"img": slice_2_data, "scan_id": '7', "z_id": 2, "is_start": False, ...}, # Position 2
        #     # ...
        #     {"img": slice_99_data, "scan_id": '7', "z_id": 99, "is_end": True, ...}   # Position 99
        # ]

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1], #Using ii:ii+1 keeps the "stack" structure with 1 slice, while ii gives you just the flat paper
                            "mean":mean,
                            "std":std,
                           "lb":lb[..., ii: ii + 1],
                           "is_start": False,
                           "is_end": False,
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last frame, note the is_end flag
            out_list.append( {"img": img[..., ii: ii + 1], # extracting the 2D scan of the 3D scan 
                            "mean":mean,
                            "std":std,
                           "lb":lb[..., ii: ii+ 1],
                           "is_start": False,
                           "is_end": True,
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1
        return out_list

    def read_classfiles(self):
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_1.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        if self.is_train:
            if len(self.exclude_lbs) > 0:
                for _ex_cls in self.exclude_lbs:
                    if curr_dict["z_id"] in self.tp1_cls_map[self.label_name[_ex_cls]][curr_dict["scan_id"]]: # this slice need to be excluded since it contains label which is supposed to be unseen
                        return self.__getitem__(index + torch.randint(low = 0, high = self.__len__() - 1, size = (1,)))

            comp = np.concatenate( [curr_dict["img"], curr_dict["lb"]], axis = -1 )
            img, lb = self.transforms(comp, c_img = 1, c_label = 1, nclass = self.nclass, use_onehot = False)

        else:
            img = curr_dict['img']
            lb = curr_dict['lb']
            # print(lb.sum(), lb.shape, np.unique(lb))

        img = np.float32(img)
        # print(img.shape)
        lb = np.float32(lb).squeeze(-1) # NOTE: to be suitable for the PANet structure
        # print(lb.sum(), lb.shape, np.unique(lb))

        img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
        lb  = torch.from_numpy( lb) #.transpose((2,0,1)))
        # print(lb.sum(), lb.shape, np.unique(lb))

        if self.tile_z_dim:
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            # lb = lb.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

        is_start = curr_dict["is_start"]
        is_end = curr_dict["is_end"]
        nframe = np.int32(curr_dict["nframe"])
        scan_id = curr_dict["scan_id"]
        z_id    = curr_dict["z_id"]
        mean, std = curr_dict["mean"], curr_dict["std"]

        sample = {"image": img,
                "label":lb,
                "is_start": is_start,
                "is_end": is_end,
                "nframe": nframe,
                "scan_id": scan_id,
                "z_id": z_id,
                "mean":mean,
                "std":std
                }
        #print(mean,std)
        # Add auxiliary attributes
        if self.aux_attrib is not None:
            for key_prefix in self.aux_attrib:
                # Process the data sample, create new attributes and save them in a dictionary
                aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
                for key_suffix in aux_attrib_val:
                    # one function may create multiple attributes, so we need suffix to distinguish them
                    sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.fix_length != None:
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def update_subclass_lookup(self):
        """
        Updating the class-slice indexing list
        Args:
            [internal] overall_slice_by_cls:
                {
                    class1: {pid1: [slice1, slice2, ....],
                                pid2: [slice1, slice2]},
                                ...}
                    class2:
                    ...
                }
        out[internal]:
                {
                    class1: [ idx1, idx2, ...  ],
                    class2: [ idx1, idx2, ...  ],
                    ...
                }

        """
        # delete previous ones if any
        assert self.overall_slice_by_cls is not None

        if not hasattr(self, 'idx_by_class'):
            self.idx_by_class = {}
        # filter the new one given the actual list
        for cls in self.label_name:
            if cls not in self.idx_by_class.keys():
                self.idx_by_class[cls] = []
            else:
                del self.idx_by_class[cls][:]
        for cls, dict_by_pid in self.overall_slice_by_cls.items():
            for pid, slice_list in dict_by_pid.items():
                if pid not in self.pid_curr_load:
                    continue
                self.idx_by_class[cls] += [ self.scan_z_idx[pid][_sli] for _sli in slice_list ]
        print("###### index-by-class table has been reloaded ######")

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask. Used when getting the support
        """
        # Dense Mask
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}

    def subsets(self, sub_args_lst=None):
        """
        Override base-class subset method
        Create subsets by scan_ids

        output: list [[<fid in each class>] <class1>, <class2>     ]
        """

        if sub_args_lst is not None:
            subsets = []
            ii = 0
            for cls_name, index_list in self.idx_by_class.items():
                subsets.append( Subset(dataset = self, indices = index_list, sub_attrib_args = sub_args_lst[ii])  )
                ii += 1
        else:
            subsets = [Subset(dataset=self, indices=index_list) for _, index_list in self.idx_by_class.items()]
        return subsets

    def get_support(self, curr_class: int, class_idx: list, scan_idx: list, npart: int):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        assert curr_class != 0; assert 0 not in class_idx
        assert not self.is_train
        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        # self.potential_support_sid = np.random.choice(self.scan_ids, replace = False, size = npart)#self.scan_per_load)# 
        # print(self.potential_support_sid)
        # print(f'###### Using {len(scan_idx)} shot evaluation!')
        print(f'###### Using {len(self.potential_support_sid)} shot evaluation!')
        # print(scan_idx, self.potential_support_sid)

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        print(f'###### Parts percentage: {pcts} ######')

        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for _scan_id in self.potential_support_sid: #scan_order in scan_idx:
                # _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                _zlist = self.tp1_cls_map[self.label_name[curr_class]][_scan_id] # list of indices
                _zid = _zlist[int(pcts[_part] * len(_zlist))]
                _glb_idx = self.scan_z_idx[_scan_id][_zid]

                # almost copy-paste __getitem__ but no augmentation
                curr_dict = self.actual_dataset[_glb_idx]
                img = curr_dict['img']
                lb = curr_dict['lb']

                img = np.float32(img)
                lb = np.float32(lb).squeeze(-1) # NOTE: to be suitable for the PANet structure

                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
                lb  = torch.from_numpy( lb) #.transpose((2,0,1)) )

                if self.tile_z_dim:
                    img = img.repeat( [ self.tile_z_dim, 1, 1] )
                    # lb = lb.repeat( [ self.tile_z_dim, 1, 1] )
                    assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                is_start    = curr_dict["is_start"]
                is_end      = curr_dict["is_end"]
                nframe      = np.int32(curr_dict["nframe"])
                scan_id     = curr_dict["scan_id"]
                z_id        = curr_dict["z_id"]

                sample = {"image": img,
                        "label":lb,
                        "is_start": is_start,
                        "inst": None,
                        "scribble": None,
                        "is_end": is_end,
                        "nframe": nframe,
                        "scan_id": scan_id,
                        "z_id": z_id
                        }

                concat_buffer.append(sample)
            # print(len(concat_buffer))
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "label": torch.stack([itm["label"] for itm in concat_buffer], dim = 0),

                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            support_mask.append(  self.getMaskMedImg( itm["label"], curr_class, class_idx  ))

        # print(len(support_images))

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
        }

