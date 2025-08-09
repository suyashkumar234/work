import numpy as np
import SimpleITK as sitk

DATASET_INFO = {
    "CHAOST2": {
        "PSEU_LABEL_NAME": ["BGD", "SUPFG"],
        "REAL_LABEL_NAME": ["BG", "LIVER", "RK", "LK", "SPLEEN"],
        "_SEP": [0, 4, 8, 12, 16, 20],
        "MODALITY": "MR",
        "LABEL_GROUP": {
            "pa_all": set(range(1, 5)),
            0: set([1, 4]),
            1: set([2, 3])
        }
    },
    "SABS": {
        "PSEU_LABEL_NAME": ["BGD", "SUPFG"],
        "REAL_LABEL_NAME": ["BGD", "SPLEEN", "RK", "LK", "GALLBLADDER", "ESOPHAGUS", "LIVER", 
  "STOMACH", "AORTA", "IVC", "PS_VEIN", "PANCREAS", "AG_R", "AG_L"],
        "_SEP": [0, 6, 12, 18, 24, 30],
        "MODALITY": "CT",
        "LABEL_GROUP": {
            "pa_all": set([1, 2, 3, 6]),
            0: set([1, 6]),
            1: set([2, 3])
        }
    }
}

def read_nii_bysitk(input_fid, peel_info=False):
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    if peel_info:
        info_obj = {
            "spacing": img_obj.GetSpacing(),
            "origin": img_obj.GetOrigin(),
            "direction": img_obj.GetDirection(),
            "array_size": img_np.shape
        }
        return img_np, info_obj
    else:
        return img_np

def get_normalize_op(modality, fids):
    def get_CT_statistics(scan_fids):
        total_val = 0
        n_pix = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_val += in_img.sum()
            n_pix += np.prod(in_img.shape)
            del in_img
        meanval = total_val / n_pix
        total_var = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2)
            del in_img
        var_all = total_var / n_pix
        global_std = var_all ** 0.5
        return meanval, global_std

    if modality == "MR":
        def MR_normalize(x_in):
            return (x_in - x_in.mean()) / x_in.std(), x_in.mean(), x_in.std()
        return MR_normalize
    elif modality == "CT":
        ct_mean, ct_std = get_CT_statistics(fids)
        def CT_normalize(x_in):
            return (x_in - ct_mean) / ct_std, ct_mean, ct_std
        return CT_normalize
