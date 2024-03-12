import os
import copy
import time 
import numpy as np
import nibabel as nib
import monai
import torch
import SimpleITK as sitk
from PIL import Image
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
import cv2
import detect_onlyget_xy
import sys
sys.path.insert(0, r"/opt/app/yolov7-main")

def resample(sitk_volume, new_spacing, new_size, default_value=0, is_label=False):
    """1) Create resampler"""
    resample = sitk.ResampleImageFilter() 
    
    """2) Set parameters"""
    #set interpolation method, output direction, default pixel value
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    resample.SetDefaultPixelValue(default_value)
    
    #set output spacing
    new_spacing = np.array(new_spacing, dtype=np.double)
    resample.SetOutputSpacing(new_spacing)
    
    #set output size and origin
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_size_no_shift = np.int16(np.ceil(old_size*old_spacing/new_spacing))
    old_origin = np.array(sitk_volume.GetOrigin())
    
    shift_amount = np.int16(np.floor((new_size_no_shift - new_size)/2))*new_spacing
    new_origin = old_origin + shift_amount
    
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        pass
    """3) execute"""
    new_volume = resample.Execute(sitk_volume)
    return new_volume

def convert_to_sitk(img_volume, spacing, origin, direction=None):
        sitk_volume = sitk.GetImageFromArray(img_volume)
        sitk_volume.SetOrigin(origin)
        sitk_volume.SetSpacing(spacing)
        if direction:
            sitk_volume.SetDirection(direction)
        return sitk_volume

def get_crop_coord(x1, y1, z1, x2, y2, z2, max_z):
        
        if x2 - x1 < 144:
            res = 144 - (x2 - x1)
            res /= 2 
            res = round(res)
            nx1 = x1 - res
            nx2 = x2 + res
        else:
            nx2 = x2
            nx1 = x1
        
        if y2 - y1 < 144:
            res = 144 - (y2 - y1)
            res /= 2
            res = round(res)
            ny1 = y1 - res
            ny2 = y2 + res
        else:
            ny2 = y2
            ny1 = y1
        
        z_minimum = 32
        if z2 - z1 < z_minimum:
            res = z_minimum - (z2 - z1)
            res /= 2
            res = round(res)
            nz1 = z1 - res
            nz2 = z2 + res
            
        else:
            nz2 = z2
            nz1 = z1
        
        nz1 = max(nz1, 0)
        nz2 = min(max_z, nz2)
        
        if nz2 - nz1 < z_minimum and nz1 <= 10:
            add_1 = nz1 - 0
            add_2 = z_minimum - (nz2- nz1)
            nz1 -= add_1
            nz2 += add_2
        
        elif nz2 - nz1 < z_minimum and nz2 >= max_z - 10:
            nz1 = nz2 - z_minimum   
            nz2 = max_z
        return nx1, ny1, nz1, nx2, ny2, nz2

def give_margin_to_bbox(bbox):
    x1, y1, z1 = bbox[0][0], bbox[0][1], bbox[0][2]
    x2, y2, z2 = bbox[1][0], bbox[1][1], bbox[1][2]
    
    xy_margin = 20
    z_margin = 10
    z2 += z_margin + 1
    z1 -= z_margin
    y2 += xy_margin + 1
    y1 -= xy_margin
    x2 += xy_margin + 1
    x1 -= xy_margin
    bbox[0][0], bbox[0][1], bbox[0][2] = x1, y1, z1
    bbox[1][0], bbox[1][1], bbox[1][2] = x2, y2, z2

    return bbox

def increase_boxsize_to_minimum(bbox, max_z, minimum_xy, minimum_z):
    x1, y1, z1 = bbox[0][0], bbox[0][1], bbox[0][2]
    x2, y2, z2 = bbox[1][0], bbox[1][1], bbox[1][2]
    if x2 - x1 < minimum_xy:
        res = minimum_xy - (x2 - x1)
        res /= 2 
        res = round(res)
        nx1 = x1 - res
        nx2 = x2 + res
    else:
        nx2 = x2
        nx1 = x1
    
    if y2 - y1 < minimum_xy:
        res = minimum_xy - (y2 - y1)
        res /= 2
        res = round(res)
        ny1 = y1 - res
        ny2 = y2 + res
    else:
        ny2 = y2
        ny1 = y1
    
    z_minimum = minimum_z
    if z2 - z1 < z_minimum:
        res = z_minimum - (z2 - z1)
        res /= 2
        res = round(res)
        nz1 = z1 - res
        nz2 = z2 + res
        
    else:
        nz2 = z2
        nz1 = z1
    
    nz1 = max(nz1, 0)
    nz2 = min(max_z, nz2)
    
    if nz2 - nz1 < z_minimum and nz1 <= 10:
        add_1 = nz1 - 0
        add_2 = z_minimum - (nz2- nz1)
        nz1 -= add_1
        nz2 += add_2
    
    elif nz2 - nz1 < z_minimum and nz2 >= max_z - 10:
        nz1 = nz2 - z_minimum   
        nz2 = max_z
    bbox[0][0], bbox[0][1], bbox[0][2] = nx1, ny1, nz1
    bbox[1][0], bbox[1][1], bbox[1][2] = nx2, ny2, nz2
    
    return bbox

def change_512_to_1024_bbox(bbox):
    x1, y1, z1 = bbox[0][0], bbox[0][1], bbox[0][2]
    x2, y2, z2 = bbox[1][0], bbox[1][1], bbox[1][2]

    x1 *= 2; y1 *= 2; x2 *= 2; y2 *= 2

    bbox[0][0], bbox[0][1], bbox[0][2] = x1, y1, z1
    bbox[1][0], bbox[1][1], bbox[1][2] = x2, y2, z2

    return bbox 

def crop_arr_by_bbox(arr, bbox):
    # array should be aligned (z, y, x) form
    x1, y1, z1 = bbox[0][0], bbox[0][1], bbox[0][2]
    x2, y2, z2 = bbox[1][0], bbox[1][1], bbox[1][2]
    crop_arr = arr[z1:z2, y1:y2, x1:x2]
    return crop_arr

def windowing_to_255(ct_img_arr):
    win_min = -1024
    win_max = 3071
    ct_img_arr[ct_img_arr < win_min] = win_min
    ct_img_arr[ct_img_arr > win_max] = win_max 
    ct_img_arr = 255 * (ct_img_arr - win_min)/(win_max - win_min)
    ct_img_arr = ct_img_arr.astype("uint8")
    return ct_img_arr

def inference_hanseg(ct_sitk, mr_sitk):
    seg_weights_0_fol_path = r"/opt/app/segmentation_weights_fold0"
    # seg_weights_1_fol_path = r"/opt/app/segmentation_weights_fold1"
    seg_weights_2_fol_path = r"/opt/app/segmentation_weights_fold2"
    seg_weights_3_fol_path = r"/opt/app/segmentation_weights_fold3"
    seg_weights_4_fol_path = r"/opt/app/segmentation_weights_fold4"
    seg_weights_5_fol_path = r"/opt/app/segmentation_weights_fold5"
    
    
    

    seg_weights_0_list = os.listdir(seg_weights_0_fol_path)
    object_detection_weights_path_1 = r"/opt/app/detection_weights/fold0.pt"
    object_detection_weights_path_2 = r"/opt/app/detection_weights/fold1.pt"
    object_detection_weights_path_3 = r"/opt/app/detection_weights/fold2.pt"
    object_detection_weights_path_4 = r"/opt/app/detection_weights/fold3.pt"
    object_detection_weights_path_5 = r"/opt/app/detection_weights/fold4.pt"
    oar_list = [ 
                "Arytenoid",
                "A_Carotid_L",
                "A_Carotid_R",
                "Bone_Mandible",
                "Brainstem",
                "BuccalMucosa",
                "Cavity_Oral",
                "Cochlea_L",
                "Cochlea_R",
                "Cricopharyngeus",
                "Esophagus_S",
                "Eye_AL",
                "Eye_AR",
                "Eye_PL",
                "Eye_PR",
                "Glnd_Lacrimal_L",
                "Glnd_Lacrimal_R",
                "Glnd_Submand_L",
                "Glnd_Submand_R",
                "Glnd_Thyroid",
                "Glottis",
                "Larynx_SG",
                "Lips",
                "OpticChiasm",
                "OpticNrv_L",
                "OpticNrv_R",
                "Parotid_L",
                "Parotid_R",
                "Pituitary",
                "SpinalCord",
                ]

    oar_dict = {k:idx for idx, k in enumerate(oar_list)}
    half_mode = 0
    if ct_sitk.GetSize()[0] <= 512:
        meta_spacing = np.array(ct_sitk.GetSpacing(), np.double)
        meta_size = np.array(ct_sitk.GetSize())
        meta_direction = ct_sitk.GetDirection()
        meta_origin = ct_sitk.GetOrigin()

        new_spacing = list(ct_sitk.GetSpacing())
        new_spacing[0] /= 2; new_spacing[1] /= 2
        new_size = list(ct_sitk.GetSize())
        new_size[0] *= 2; new_size[1] *= 2
        ct_sitk = resample(ct_sitk, new_spacing, new_size,-1024)
        mr_sitk = resample(mr_sitk, new_spacing, new_size, 0)
        half_mode = 1

    else:
        meta_spacing = np.array(ct_sitk.GetSpacing(), np.double)
        meta_size = np.array(ct_sitk.GetSize())
        meta_direction = ct_sitk.GetDirection()
        meta_origin = ct_sitk.GetOrigin()
        new_spacing = list(ct_sitk.GetSpacing())
        new_size = list(ct_sitk.GetSize())
        pass
    
    if ct_sitk.GetSize()[0] <= 512:
        ct_obj_det_sitk = copy.deepcopy(ct_sitk)
    else:
        
        obj_det_spacing = list(ct_sitk.GetSpacing())
        obj_det_size = list(ct_sitk.GetSize())

        obj_det_spacing[0] *= 2; obj_det_spacing[1] *= 2
        obj_det_size[0] /= 2; obj_det_size[1] /= 2

        ct_obj_det_sitk = resample(ct_sitk, obj_det_spacing, obj_det_size, -1024)

    ct_arr = sitk.GetArrayFromImage(ct_sitk)
    ct_obj_arr = sitk.GetArrayFromImage(ct_obj_det_sitk)

    mr_arr = sitk.GetArrayFromImage(mr_sitk)

    ct_img_arr = np.copy(ct_obj_arr).astype("float64")
    ct_img_arr = windowing_to_255(ct_img_arr)

    npy_coord_list_1 = detect_onlyget_xy.detect(ct_img_arr, object_detection_weights_path_1, detect_conf= 0.10, save_img=False)
    npy_coord_list_2 = detect_onlyget_xy.detect(ct_img_arr, object_detection_weights_path_2, detect_conf= 0.11, save_img=False)
    npy_coord_list_3 = detect_onlyget_xy.detect(ct_img_arr, object_detection_weights_path_3, detect_conf= 0.12, save_img=False)
    npy_coord_list_4 = detect_onlyget_xy.detect(ct_img_arr, object_detection_weights_path_4, detect_conf= 0.13, save_img=False)
    npy_coord_list_5 = detect_onlyget_xy.detect(ct_img_arr, object_detection_weights_path_5, detect_conf= 0.14, save_img=False)
    # 0번째 축이 z축 순서, for문으로 순회할 시 각 slice에 있는 oar을 알 수있음. 그리고 이건 리스트임
    # print(npy_coord_list)
    def get_oar_end_check(npy_coord_list):
        oar_check = {}
        oar_end_check = {}
        for j in range(len(oar_list)):
            oar_check[j] = []
            oar_end_check[j] = []
        
        zdx_num = 0
        for i in range(len(npy_coord_list)):
            coord_npy = npy_coord_list[i]
            # i 가 결국 z 번째인듯?
            # coord_npy[j] = [x1, y1, x2, y2, class]

            for j in range(len(coord_npy)):
                coord_list = [coord_npy[j][0], coord_npy[j][1], coord_npy[j][2], coord_npy[j][3], zdx_num]
                
                oar_check[coord_npy[j][4]].append(coord_list)
                
            zdx_num += 1

        #여기가 class 있는지 없는지 체크하는 부분.
        #없을시 그 부분만 제외하는거 필요할 듯.
        for oar_num in range(len(oar_check)):
            if len(oar_check[oar_num]) == 0:
                print(oar_num)
        
        exception_dict = {}
        for oar_num in range(len(oar_check)):
            if len(oar_check[oar_num]) == 0: # 없는 부분 제외하는 코드.
                exception_dict[oar_num] = 1
                continue
            
            temp_arr = np.array(oar_check[oar_num])
            # print("dlrjs", temp_arr)
            x1_arr = temp_arr[:, 0]
            y1_arr = temp_arr[:, 1]
            x2_arr = temp_arr[:, 2]
            y2_arr = temp_arr[:, 3]
            z_arr = temp_arr[:, 4]
            x1_sort_arr = np.sort(x1_arr)
            y1_sort_arr = np.sort(y1_arr)
            x2_sort_arr = np.sort(x2_arr)
            y2_sort_arr = np.sort(y2_arr)   
            # 95퍼센트로 잡기


            x1_idx = int(len(x1_sort_arr) * 0.05)
            y1_idx = int(len(y1_sort_arr) * 0.05)
            x2_idx = int(len(x2_sort_arr) * 0.95)
            y2_idx = int(len(y2_sort_arr) * 0.95)
            

            x1 = x1_sort_arr[x1_idx]
            y1 = y1_sort_arr[y1_idx]
            x2 = x2_sort_arr[x2_idx]
            y2 = y2_sort_arr[y2_idx]
            if len(z_arr) < 10:
                z1 = z_arr[0]
                z2 = z_arr[-1]
            else:
                z1 = z_arr[3]
                z2 = z_arr[-4]
            oar_end_check[oar_num] = [[x1, y1, z1], [x2, y2, z2]]
        return oar_end_check, exception_dict
    
    oar_end_check_1, exception_dict_1 = get_oar_end_check(npy_coord_list_1)
    oar_end_check_2, exception_dict_2 = get_oar_end_check(npy_coord_list_2)
    oar_end_check_3, exception_dict_3 = get_oar_end_check(npy_coord_list_3)
    oar_end_check_4, exception_dict_4 = get_oar_end_check(npy_coord_list_4)
    oar_end_check_5, exception_dict_5 = get_oar_end_check(npy_coord_list_5)

    oar_end_check = {}
    for kv1, kv2, kv3, kv4, kv5 in zip(oar_end_check_1.items(), oar_end_check_2.items(), oar_end_check_3.items(), oar_end_check_4.items(), oar_end_check_5.items()):
        key = kv1[0]
        val_1 = list(kv1)[1]; val_2 = list(kv2)[1]; val_3 = list(kv3)[1]
        val_4 = list(kv4)[1]; val_5 = list(kv5)[1]
        # print(val_1)
        # print(val_2)
        # print(val_3)
        x1_median = np.uint16(np.median([val_1[0][0], val_2[0][0], val_3[0][0], val_4[0][0], val_5[0][0]]))
        x2_median = np.uint16(np.median([val_1[1][0], val_2[1][0], val_3[1][0], val_4[1][0], val_5[1][0]]))
        y1_median = np.uint16(np.median([val_1[0][1], val_2[0][1], val_3[0][1], val_4[0][1] ,val_5[0][1]]))
        y2_median = np.uint16(np.median([val_1[1][1], val_2[1][1], val_3[1][1], val_4[1][1], val_5[1][1]]))
        z1_median = np.uint16(np.median([val_1[0][2], val_2[0][2], val_3[0][2], val_4[0][2], val_5[0][2]]))
        z2_median = np.uint16(np.median([val_1[1][2], val_2[1][2], val_3[1][2], val_4[1][2], val_5[1][2]]))
        # print(kv1)
        # print(kv2)
        # print(kv3)
        oar_end_check[key] = [[x1_median, y1_median, z1_median], [x2_median, y2_median, z2_median]]
    
    exception_dict = {}
    for kv1, kv2, kv3, kv4, kv5 in zip(exception_dict_1.items(), 
                                       exception_dict_2.items(), 
                                       exception_dict_3.items(), 
                                       exception_dict_4.items(), 
                                       exception_dict_5.items()):
        key = kv1[0]
        val_1 = kv1[1]; val_2 = kv2[1]; val_3 = kv3[1]; val_4 = kv4[1]; val_5 = kv5[1]
        val_sum = val_1 + val_2 + val_3 + val_4 + val_5
        if val_sum >= 3:
            val_sum = 1
        else:
            val_sum = 0
        exception_dict[key] = val_sum
        
    oar_label_dict = {'Arytenoid': 0, 'A_Carotid_L': 1, 'A_Carotid_R': 2, 'Bone_Mandible': 3, 'Brainstem': 4, 'BuccalMucosa': 5, 
           'Cavity_Oral': 6, 'Cochlea_L': 7, 'Cochlea_R': 8, 'Cricopharyngeus': 9, 'Esophagus_S': 10, 
           'Eye_AL': 11, 'Eye_AR': 12, 'Eye_PL': 13, 'Eye_PR': 14, 'Glnd_Lacrimal_L': 15, 
           'Glnd_Lacrimal_R': 16, 'Glnd_Submand_L': 17, 'Glnd_Submand_R': 18, 'Glnd_Thyroid': 19, 'Glottis': 20, 
           'Larynx_SG': 21, 'Lips': 22, 'OpticChiasm': 23, 'OpticNrv_L': 24, 'OpticNrv_R': 25, 
           'Parotid_L': 26, 'Parotid_R': 27, 'Pituitary': 28, 'SpinalCord': 29}
    oar_label_dict_rev = {v:k for k,v in oar_label_dict.items()}

    # oar_label_dict = {'Arytenoid': 0, 'A_Carotid_L': 1, 'A_Carotid_R': 2, 'Bone_Mandible': 3, 'Brainstem': 4, 'BuccalMucosa': 5, 
    #        'Cavity_Oral': 6, 'Cochlea_L': 7, 'Cochlea_R': 8, 'Cricopharyngeus': 9, 'Esophagus_S': 10, 
    #        'Eye_AL': 11, 'Eye_AR': 12, 'Eye_PL': 13, 'Eye_PR': 14, 'Glnd_Lacrimal_L': 15, 
    #        'Glnd_Lacrimal_R': 16, 'Glnd_Submand_L': 17, 'Glnd_Submand_R': 18, 'Glnd_Thyroid': 19, 'Glottis': 20, 
    #        'Larynx_SG': 21, 'Lips': 22, 'OpticChiasm': 23, 'OpticNrv_L': 24}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks_fold0 = [[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    st_fold0 = [[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
    uks_fold0 = st_fold0[1:]
    OAR_nums_plus_one = 2
    model_fold_0 = DynUNet( # 96 96 32
        spatial_dims=3,
        in_channels=5,
        out_channels=OAR_nums_plus_one,
        kernel_size=ks_fold0,
        strides=st_fold0,
        upsample_kernel_size=uks_fold0,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
    # model_fold_1 = DynUNet( # 96 96 32
    #     spatial_dims=3,
    #     in_channels=5,
    #     out_channels=OAR_nums_plus_one,
    #     kernel_size=ks_fold0,
    #     strides=st_fold0,
    #     upsample_kernel_size=uks_fold0,
    #     act_name= "LEAKYRELU",
    #     deep_supervision=False
    # ).to(device)
    model_fold_2 = DynUNet( # 96 96 32
        spatial_dims=3,
        in_channels=5,
        out_channels=OAR_nums_plus_one,
        kernel_size=ks_fold0,
        strides=st_fold0,
        upsample_kernel_size=uks_fold0,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
    model_fold_4 = DynUNet( # 96 96 32
        spatial_dims=3,
        in_channels=5,
        out_channels=OAR_nums_plus_one,
        kernel_size=ks_fold0,
        strides=st_fold0,
        upsample_kernel_size=uks_fold0,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
    model_fold_5 = DynUNet( # 96 96 32
        spatial_dims=3,
        in_channels=5,
        out_channels=OAR_nums_plus_one,
        kernel_size=ks_fold0,
        strides=st_fold0,
        upsample_kernel_size=uks_fold0,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
    model_fold_3 = DynUNet(
        spatial_dims=3,
        in_channels=6,
        out_channels=OAR_nums_plus_one,
        kernel_size=ks_fold0,
        strides=st_fold0,
        upsample_kernel_size=uks_fold0,
        dropout=0.3,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)

    pred_list = []
    oar_crop_meta_dict = {}
    
    
    for oar, bbox in oar_end_check.items():
        if oar == 0:
            continue
        if oar in exception_dict: #만약 object detection 단계에서 탐지 되지 않았다면,,,
            continue
        R_conv_mode = 0
        if "_R" in oar_label_dict_rev[oar] or "AR" in oar_label_dict_rev[oar] or "PR" in oar_label_dict_rev[oar]:
            R_conv_mode = 1 
        print(oar_label_dict_rev[oar])
        secpptime = time.time()

        # print(oar_name)
        
        
        
        nii_im_arr = np.copy(ct_arr) # z y x
        nii_im_arr = np.transpose(nii_im_arr, (2, 1, 0)) # x y z 
        
        nii_mr_arr = np.copy(mr_arr)
        nii_mr_arr = np.transpose(nii_mr_arr, (2, 1, 0))

        x1, y1, z1 = bbox[0][0], bbox[0][1], bbox[0][2]
        x2, y2, z2 = bbox[1][0], bbox[1][1], bbox[1][2]
        

        xy_margin = 20
        z_margin = 10
        z2 += z_margin + 1
        z1 -= z_margin
        y2 += xy_margin + 1
        y1 -= xy_margin
        x2 += xy_margin + 1
        x1 -= xy_margin

        coord_list = [[x1, y1, z1], [x2, y2, z2]]
        # if oar_123[oar_name] == oar :
            
        # .............................. 
        # .......'''''''''''''''........
        nx1, ny1, nz1, nx2, ny2, nz2 = get_crop_coord(x1, y1, z1, x2, y2, z2, nii_im_arr.shape[2])
        
        if nx1 <= 0:nx1 = 0
        if ny1 <= 0:ny1 = 0
        if nz1 <= 0:nz1 = 0
        if nx2 >= ct_img_arr.shape[2]:nx2 = ct_img_arr.shape[2]
        if ny2 >= ct_img_arr.shape[1]:ny2 = ct_img_arr.shape[1]
        if nz2 >= ct_img_arr.shape[0]:nz2 = ct_img_arr.shape[0]
        
        # if meta_size[0] > 512:
        nx1 *= 2; ny1 *= 2; nx2 *= 2; ny2 *= 2 # for 1024
        # else:pass

        oar_crop_meta_dict[oar] = [[nx1, ny1, nz1], [nx2, ny2, nz2]]
        
        #for arytenoid
        if oar == 3:
            x_center = (nx1 + nx2)//2 
            y_center = (ny1 + ny2)//2
            z_center = (nz1 + nz2)//2
            x_center += 4
            y_center += 110 
            z_center -= 33
            ary_x1 = x_center - 144
            ary_x2 = x_center + 144 
            ary_y1 = y_center - 144 
            ary_y2 = y_center + 144 
            ary_z1 = z_center - 32
            ary_z2 = z_center + 32 
            if ary_x1 < 0: ary_x1 = 0
            if ary_x2 > 1023: ary_x2 = 1023
            if ary_y1 < 0: ary_y1 = 0
            if ary_y2 > 1023: ary_y2 = 1023
            if ary_z1 < 0: ary_z1 = 0
            if ary_z2 >= nii_im_arr.shape[2]: ary_z2 = nii_im_arr.shape[2]-1
            ary_crop_meta = [[ary_x1, ary_y1, ary_z1], [ary_x2, ary_y2, ary_z2]]


        crop_nii_im_arr = nii_im_arr[nx1:nx2, ny1:ny2, nz1:nz2].astype("float32")
        crop_nii_mr_arr = nii_mr_arr[nx1:nx2, ny1:ny2, nz1:nz2].astype("float32")

        if R_conv_mode:
            crop_nii_im_arr_flip = np.flip(crop_nii_im_arr, axis = 0).astype("float32")
            crop_nii_mr_arr_flip = np.flip(crop_nii_mr_arr, axis = 0).astype("float32")

        crop_metadata = [[nx1, ny1, nz1], [nx2, ny2, nz2]]
        crop_nii_im_ten = torch.unsqueeze(torch.from_numpy(crop_nii_im_arr).to(device), 0)
        crop_nii_im_ten = torch.unsqueeze(crop_nii_im_ten, 0)
        crop_nii_mr_ten = torch.unsqueeze(torch.from_numpy(crop_nii_mr_arr).to(device), 0)
        crop_nii_mr_ten = torch.unsqueeze(crop_nii_mr_ten, 0)

        bone_min = -1000
        bone_max = 2000
        soft_min = -160
        soft_max = 350
        brain_min = -5
        brain_max = 65
        stroke_min = 15
        stroke_max = 45

        x1 = crop_nii_im_ten.clone().detach()
        x2 = crop_nii_im_ten.clone().detach()
        x3 = crop_nii_im_ten.clone().detach()
        x4 = crop_nii_im_ten.clone().detach()

        x1[x1<bone_min] = bone_min
        x1[x1>bone_max] = bone_max
        x1 = (x1-bone_min)/(bone_max-bone_min)
        x2[x2<soft_min] = soft_min
        x2[x2>soft_max] = soft_max
        x2 = (x2-soft_min)/(soft_max-soft_min)
        x3[x3<brain_min] = brain_min
        x3[x3>brain_max] = brain_max 
        x3 = (x3-brain_min)/(brain_max-brain_min)
        x4[x4<stroke_min] = stroke_min
        x4[x4>stroke_max] = stroke_max
        x4 = (x4 - stroke_min)/(stroke_max - stroke_min)

        val_inputs = torch.cat((crop_nii_im_ten, x1, x2, x3, x4), 1)
        val_inputs_mr = torch.cat((crop_nii_im_ten, x1, x2, x3, x4, crop_nii_mr_ten), 1)
        if R_conv_mode:
            crop_nii_im_ten_flip = torch.unsqueeze(torch.from_numpy(crop_nii_im_arr_flip).to(device), 0)
            crop_nii_im_ten_flip = torch.unsqueeze(crop_nii_im_ten_flip, 0)
            crop_nii_mr_ten_flip = torch.unsqueeze(torch.from_numpy(crop_nii_mr_arr_flip).to(device), 0)
            crop_nii_mr_ten_flip = torch.unsqueeze(crop_nii_mr_ten_flip, 0)
            bone_min = -1000
            bone_max = 2000
            soft_min = -160
            soft_max = 350
            brain_min = -5
            brain_max = 65
            stroke_min = 15
            stroke_max = 45

            x1_flip = crop_nii_im_ten_flip.clone().detach()
            x2_flip = crop_nii_im_ten_flip.clone().detach()
            x3_flip = crop_nii_im_ten_flip.clone().detach()
            x4_flip = crop_nii_im_ten_flip.clone().detach()

            x1_flip[x1_flip<bone_min] = bone_min
            x1_flip[x1_flip>bone_max] = bone_max
            x1_flip = (x1_flip-bone_min)/(bone_max-bone_min)
            x2_flip[x2_flip<soft_min] = soft_min
            x2_flip[x2_flip>soft_max] = soft_max
            x2_flip = (x2_flip-soft_min)/(soft_max-soft_min)
            x3_flip[x3_flip<brain_min] = brain_min
            x3_flip[x3_flip>brain_max] = brain_max 
            x3_flip = (x3_flip-brain_min)/(brain_max-brain_min)
            x4_flip[x4_flip<stroke_min] = stroke_min
            x4_flip[x4_flip>stroke_max] = stroke_max
            x4_flip = (x4_flip - stroke_min)/(stroke_max - stroke_min)
            val_inputs_flip = torch.cat((crop_nii_im_ten_flip, x1_flip, x2_flip, x3_flip, x4_flip), 1)
            val_inputs_flip_mr = torch.cat((crop_nii_im_ten_flip, x1_flip, x2_flip, x3_flip, x4_flip, crop_nii_mr_ten_flip), 1)
        
        
        model_fold_0.load_state_dict(torch.load(os.path.join(seg_weights_0_fol_path, oar_label_dict_rev[oar].lower() + "_model_1024.pth")))
        if R_conv_mode:
            # model_fold_1.load_state_dict(torch.load(os.path.join(seg_weights_1_fol_path, oar_label_dict_rev[oar].lower()[:-1] + "l" + "_model_1024_fold1.pth")))
            model_fold_2.load_state_dict(torch.load(os.path.join(seg_weights_2_fol_path, oar_label_dict_rev[oar].lower()[:-1] + "l" + "_model_1024_fold1.pth")))
            model_fold_3.load_state_dict(torch.load(os.path.join(seg_weights_3_fol_path, oar_label_dict_rev[oar].lower()[:-1] + "l" + "_model_1024_fold1.pth")))
            model_fold_4.load_state_dict(torch.load(os.path.join(seg_weights_4_fol_path, oar_label_dict_rev[oar].lower()[:-1] + "l" + "_model_1024_fold4.pth")))
            model_fold_5.load_state_dict(torch.load(os.path.join(seg_weights_5_fol_path, oar_label_dict_rev[oar].lower()[:-1] + "l" + "_model_1024_fold5.pth")))
        else:
            # model_fold_1.load_state_dict(torch.load(os.path.join(seg_weights_1_fol_path, oar_label_dict_rev[oar].lower() + "_model_1024_fold1.pth"))) #need to model_fold_1 ordering #need to model_fold_0 ordering
            model_fold_2.load_state_dict(torch.load(os.path.join(seg_weights_2_fol_path, oar_label_dict_rev[oar].lower() + "_model_1024_fold1.pth"))) #need to model_fold_1 ordering #need to model_fold_0 ordering
            model_fold_3.load_state_dict(torch.load(os.path.join(seg_weights_3_fol_path, oar_label_dict_rev[oar].lower() + "_model_1024_fold1.pth")))
            model_fold_4.load_state_dict(torch.load(os.path.join(seg_weights_4_fol_path, oar_label_dict_rev[oar].lower() + "_model_1024_fold4.pth")))
            model_fold_5.load_state_dict(torch.load(os.path.join(seg_weights_5_fol_path, oar_label_dict_rev[oar].lower() + "_model_1024_fold5.pth")))
        model_fold_0.eval()
        # model_fold_1.eval()
        model_fold_2.eval()
        model_fold_3.eval()
        model_fold_4.eval()
        model_fold_5.eval()

        # model_fold_3.eval()
        with torch.no_grad():
            
            val_outputs_fold0 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_0, overlap=0.25)
            if R_conv_mode:
                # val_outputs_fold1 = sliding_window_inference(val_inputs_flip, (192, 192, 32), 4, model_fold_1, overlap=0.40)
                val_outputs_fold2 = sliding_window_inference(val_inputs_flip, (192, 192, 32), 4, model_fold_2, overlap=0.25)
                val_outputs_fold3 = sliding_window_inference(val_inputs_flip_mr, (192, 192, 32), 4, model_fold_3, overlap=0.25)
                val_outputs_fold4 = sliding_window_inference(val_inputs_flip, (192, 192, 32), 4, model_fold_4, overlap=0.25)
                val_outputs_fold5 = sliding_window_inference(val_inputs_flip, (192, 192, 32), 4, model_fold_5, overlap=0.25)
            else:
                # val_outputs_fold1 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_1, overlap=0.40)
                val_outputs_fold2 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_2, overlap=0.25)
                val_outputs_fold3 = sliding_window_inference(val_inputs_mr, (192, 192, 32), 4, model_fold_3, overlap=0.25)
                val_outputs_fold4 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_4, overlap=0.25)
                val_outputs_fold5 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_5, overlap=0.25)
            last_outputs_fold0 = torch.argmax(val_outputs_fold0, dim=1).detach().cpu()[0].numpy()
            # last_outputs_fold1 = torch.argmax(val_outputs_fold1, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold2 = torch.argmax(val_outputs_fold2, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold3 = torch.argmax(val_outputs_fold3, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold4 = torch.argmax(val_outputs_fold4, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold5 = torch.argmax(val_outputs_fold5, dim=1).detach().cpu()[0].numpy()

        if R_conv_mode:
            # last_outputs_fold1 = np.flip(last_outputs_fold1, axis = 0).astype("float32")
            last_outputs_fold2 = np.flip(last_outputs_fold2, axis = 0).astype("float32")
            last_outputs_fold3 = np.flip(last_outputs_fold3, axis = 0).astype("float32")
            last_outputs_fold4 = np.flip(last_outputs_fold4, axis = 0).astype("float32")
            last_outputs_fold5 = np.flip(last_outputs_fold5, axis = 0).astype("float32")
        if half_mode == 0:
            target_spacing = [float(ct_sitk.GetSpacing()[0]), float(ct_sitk.GetSpacing()[1]), float(ct_sitk.GetSpacing()[2])]
        elif half_mode == 1:
            target_spacing = meta_spacing
        
        last_outputs_fold0[last_outputs_fold0 < 0.5] = 0
        last_outputs_fold0[last_outputs_fold0 >=0.5] = 1
        # last_outputs_fold1[last_outputs_fold1 < 0.5] = 0
        # last_outputs_fold1[last_outputs_fold1 >=0.5] = 1
        last_outputs_fold2[last_outputs_fold2 < 0.5] = 0
        last_outputs_fold2[last_outputs_fold2 >=0.5] = 1
        last_outputs_fold3[last_outputs_fold3 < 0.5] = 0
        last_outputs_fold3[last_outputs_fold3 >=0.5] = 1
        last_outputs_fold4[last_outputs_fold4 < 0.5] = 0
        last_outputs_fold4[last_outputs_fold4 >=0.5] = 1
        last_outputs_fold5[last_outputs_fold5 < 0.5] = 0
        last_outputs_fold5[last_outputs_fold5 >=0.5] = 1

        if meta_size[0] >= 513:
            recon_size = [last_outputs_fold0.shape[0], last_outputs_fold0.shape[1], last_outputs_fold0.shape[2]]
        else:
            recon_size = [last_outputs_fold0.shape[0]/2, last_outputs_fold0.shape[1]/2, last_outputs_fold0.shape[2]]
        last_outputs_fold0 = np.transpose(last_outputs_fold0, (2, 1, 0)) # z y x
        # last_outputs_fold1 = np.transpose(last_outputs_fold1, (2, 1, 0))
        last_outputs_fold2 = np.transpose(last_outputs_fold2, (2, 1, 0))
        last_outputs_fold3 = np.transpose(last_outputs_fold3, (2, 1, 0))
        last_outputs_fold4 = np.transpose(last_outputs_fold4, (2, 1, 0))
        last_outputs_fold5 = np.transpose(last_outputs_fold5, (2, 1, 0))

        last_outputs_fold0 = last_outputs_fold0  + last_outputs_fold2 + last_outputs_fold3 + last_outputs_fold4 + last_outputs_fold5
        last_outputs_fold0 = np.where(last_outputs_fold0 >= 3, 1, 0)

        # debug_path = r"D:\!HaN_Challenge\HanSeg2023Algorithm-master\HanSeg2023Algorithm-master_local_testing_fold1\debug_img"
        # img_arr = np.copy(crop_nii_im_arr)
        # img_arr= np.transpose(img_arr, (2, 1, 0))
        # img_arr = np.stack((img_arr, ) * 3, axis = -1)
        # win_min = -150
        # win_max = 360
        # img_arr[img_arr < win_min] = win_min 
        # img_arr[img_arr > win_max] = win_max 
        # img_arr = 255 * (img_arr - win_min)/(win_max - win_min)
        # img_arr = img_arr.astype("uint8")
        
        # la_coord = np.where(last_outputs_fold0 == 1)
        # img_arr[la_coord[0], la_coord[1], la_coord[2]] = [255, 0, 0]
        # for idx in range(len(img_arr)):
        #     img = Image.fromarray(img_arr[idx])
        #     img.save(os.path.join(r"D:\!HaN_Challenge\HanSeg2023Algorithm-master\HanSeg2023Algorithm-master_local_testing_fold1\debug_img", "oar_%s.png" %(oar_label_dict_rev[oar])))


        
        last_outputs_sitk = convert_to_sitk(last_outputs_fold0, new_spacing, ct_sitk.GetOrigin()) # x y z
        last_outputs_sitk = resample(last_outputs_sitk, target_spacing, recon_size)
        last_outputs = np.transpose(sitk.GetArrayFromImage(last_outputs_sitk), (2, 1, 0)) # x y z
        last_outputs[last_outputs < 0.5] = 0
        last_outputs[last_outputs >=0.5] = 1

        # la_im =np.transpose(last_outputs, (2,1,0)).astype("uint8") * 100
        # for id in range(len(la_im)):
        #     img = Image.fromarray(la_im[id])
        #     img.save(os.path.join(r"D:\!HaN_Challenge\han_new_code\infer\testimg", "P%03d_%03d.png" %(oar, id)))
        crop_metadata_arr = np.array(crop_metadata, dtype=np.uint16)

        # if ct_sitk.GetSize()[0] <= 512:
        #     pass
        # else:
        #     crop_metadata_arr[:, :2] *= 2
        
        crop_metadata_arr_round = crop_metadata_arr.astype("uint16")
        
        posttime = time.time()
        print(posttime - secpptime)

        bs = crop_metadata_arr_round[0]
        be = crop_metadata_arr_round[1]
        
        #before transpose
        z_len = round(nii_im_arr.shape[2])
        
        # 이미지 사이즈가 32 이하일때 오류 발생. 검은색으로 filling해주는 과정 필요할듯
        # 근데 그런이미지 없을듯 해서 일단은 그대로
       
        if meta_size[0] <= 512:
            prd_la_arr_3d = np.pad(last_outputs, ((bs[0]//2, 512-be[0]//2), (bs[1]//2, 512-be[1]//2), (bs[2], z_len-be[2])))

        else:
            prd_la_arr_3d = np.pad(last_outputs, ((bs[0], 1024-be[0]), (bs[1], 1024-be[1]), (bs[2], z_len-be[2])))
        res_prd_la_arr = prd_la_arr_3d.astype('uint8')
        res_prd_la_arr = np.transpose(res_prd_la_arr, (2, 1, 0)) # z y x
        prediction_label_sitk = sitk.GetImageFromArray(res_prd_la_arr)
        if half_mode == 0:
            prediction_label_sitk.SetDirection(ct_sitk.GetDirection())
            prediction_label_sitk.SetSpacing(ct_sitk.GetSpacing())
            prediction_label_sitk.SetOrigin(ct_sitk.GetOrigin())
        elif half_mode == 1:
            prediction_label_sitk.SetDirection(meta_direction)
            prediction_label_sitk.SetSpacing(meta_spacing)
            prediction_label_sitk.SetOrigin(meta_origin)

        # red_prd_la_arr = np.transpose(res_prd_la_arr, (2, 1, 0)).astype("uint8")
        # print(red_prd_la_arr.shape)
        # crop_res_prd_la_arr = res_prd_la_arr[int(x_pad_1):int(x_pad_1+nii_size[0]), int(y_pad_1):int(y_pad_1 + nii_size[1]), 0:int(nii_size[2])]
        # crop_res_prd_la_arr = np.transpose(crop_res_prd_la_arr, (2, 1, 0)) # z y x
        # sitk.WriteImage(prediction_label_sitk, os.path.join(r"D:\!HaN_Challenge\HanSeg2023Algorithm-master\HanSeg2023Algorithm-master_local_testing_fold1\test_result", "P%s.nii.gz" %(oar_label_dict_rev[oar])))
        pred_list.append((oar_label_dict_rev[oar], prediction_label_sitk))

    # for arytenoid 
    nx1, ny1, nz1 = ary_crop_meta[0]
    nx2, ny2, nz2 = ary_crop_meta[1]
    crop_nii_im_arr = nii_im_arr[nx1:nx2, ny1:ny2, nz1:nz2].astype("float32")
    crop_nii_mr_arr = nii_mr_arr[nx1:nx2, ny1:ny2, nz1:nz2].astype("float32")
    # ## debug 
    # debug_im = np.transpose(crop_nii_im_arr, (2, 1, 0))
    # win_min = -160
    # win_max = 350 
    # debug_im[debug_im < win_min] = win_min 
    # debug_im[debug_im > win_max] = win_max 
    # debug_im = 255 * (debug_im - win_min)/(win_max - win_min)
    # debug_im = debug_im.astype("uint8")
    # for i in range(len(debug_im)):
    #     img = Image.fromarray(debug_im[i])
    #     img.save(os.path.join(r"D:\!HaN_Challenge\!ForArytenoid\debug_aryte", "%03d.png" %i))
    crop_nii_im_ten = torch.unsqueeze(torch.from_numpy(crop_nii_im_arr).to(device), 0)
    crop_nii_im_ten = torch.unsqueeze(crop_nii_im_ten, 0)
    crop_nii_mr_ten = torch.unsqueeze(torch.from_numpy(crop_nii_mr_arr).to(device), 0)
    crop_nii_mr_ten = torch.unsqueeze(crop_nii_mr_ten, 0)
    bone_min = -1000
    bone_max = 2000
    soft_min = -160
    soft_max = 350
    brain_min = -5
    brain_max = 65
    stroke_min = 15
    stroke_max = 45
    x1 = crop_nii_im_ten.clone().detach()
    x2 = crop_nii_im_ten.clone().detach()
    x3 = crop_nii_im_ten.clone().detach()
    x4 = crop_nii_im_ten.clone().detach()
    x1[x1<bone_min] = bone_min
    x1[x1>bone_max] = bone_max
    x1 = (x1-bone_min)/(bone_max-bone_min)
    x2[x2<soft_min] = soft_min
    x2[x2>soft_max] = soft_max
    x2 = (x2-soft_min)/(soft_max-soft_min)
    x3[x3<brain_min] = brain_min
    x3[x3>brain_max] = brain_max 
    x3 = (x3-brain_min)/(brain_max-brain_min)
    x4[x4<stroke_min] = stroke_min
    x4[x4>stroke_max] = stroke_max
    x4 = (x4 - stroke_min)/(stroke_max - stroke_min)
    val_inputs = torch.cat((crop_nii_im_ten, x1, x2, x3, x4), 1)
    val_inputs_mr = torch.cat((crop_nii_im_ten, x1, x2, x3, x4, crop_nii_im_ten), 1)
    model_fold_0.load_state_dict(torch.load(os.path.join(seg_weights_0_fol_path, oar_label_dict_rev[0].lower() + "_model_1024.pth")))
    # model_fold_1.load_state_dict(torch.load(os.path.join(seg_weights_1_fol_path, oar_label_dict_rev[0].lower() + "_model_1024_fold1.pth"))) #need to model_fold_1 ordering #need to model_fold_0 ordering
    model_fold_2.load_state_dict(torch.load(os.path.join(seg_weights_2_fol_path, oar_label_dict_rev[0].lower() + "_model_1024_fold1.pth"))) #need to model_fold_1 ordering #need to model_fold_0 ordering
    model_fold_3.load_state_dict(torch.load(os.path.join(seg_weights_3_fol_path, oar_label_dict_rev[0].lower() + "_model_1024_fold1.pth")))
    model_fold_4.load_state_dict(torch.load(os.path.join(seg_weights_4_fol_path, oar_label_dict_rev[0].lower() + "_model_1024_fold4.pth")))
    model_fold_5.load_state_dict(torch.load(os.path.join(seg_weights_5_fol_path, oar_label_dict_rev[0].lower() + "_model_1024_fold5.pth")))
    model_fold_0.eval()
    # model_fold_1.eval()
    model_fold_2.eval()
    model_fold_3.eval()
    model_fold_4.eval()
    model_fold_5.eval()
    with torch.no_grad():
        val_outputs_fold0 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_0, overlap=0.25)
        # val_outputs_fold1 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_1, overlap=0.40)
        val_outputs_fold2 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_2, overlap=0.25)
        val_outputs_fold3 = sliding_window_inference(val_inputs_mr, (192, 192, 32), 4, model_fold_3, overlap=0.25)
        val_outputs_fold4 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_4, overlap=0.25)
        val_outputs_fold5 = sliding_window_inference(val_inputs, (192, 192, 32), 4, model_fold_5, overlap=0.25)
        last_outputs_fold0 = torch.argmax(val_outputs_fold0, dim=1).detach().cpu()[0].numpy()
        # last_outputs_fold1 = torch.argmax(val_outputs_fold1, dim=1).detach().cpu()[0].numpy()
        last_outputs_fold2 = torch.argmax(val_outputs_fold2, dim=1).detach().cpu()[0].numpy()
        last_outputs_fold3 = torch.argmax(val_outputs_fold3, dim=1).detach().cpu()[0].numpy()
        last_outputs_fold4 = torch.argmax(val_outputs_fold4, dim=1).detach().cpu()[0].numpy()
        last_outputs_fold5 = torch.argmax(val_outputs_fold5, dim=1).detach().cpu()[0].numpy()
    if half_mode == 0:
        target_spacing = [float(ct_sitk.GetSpacing()[0]), float(ct_sitk.GetSpacing()[1]), float(ct_sitk.GetSpacing()[2])]
    elif half_mode == 1:
        target_spacing = meta_spacing
    last_outputs_fold0[last_outputs_fold0 < 0.5] = 0
    last_outputs_fold0[last_outputs_fold0 >=0.5] = 1
    # last_outputs_fold1[last_outputs_fold1 < 0.5] = 0
    # last_outputs_fold1[last_outputs_fold1 >=0.5] = 1
    last_outputs_fold2[last_outputs_fold2 < 0.5] = 0
    last_outputs_fold2[last_outputs_fold2 >=0.5] = 1
    last_outputs_fold3[last_outputs_fold3 < 0.5] = 0
    last_outputs_fold3[last_outputs_fold3 >=0.5] = 1
    last_outputs_fold4[last_outputs_fold4 < 0.5] = 0
    last_outputs_fold4[last_outputs_fold4 >=0.5] = 1
    last_outputs_fold5[last_outputs_fold5 < 0.5] = 0
    last_outputs_fold5[last_outputs_fold5 >=0.5] = 1
    if meta_size[0] >= 513:
        recon_size = [last_outputs_fold0.shape[0], last_outputs_fold0.shape[1], last_outputs_fold0.shape[2]]
    else:
        recon_size = [last_outputs_fold0.shape[0]/2, last_outputs_fold0.shape[1]/2, last_outputs_fold0.shape[2]]
    last_outputs_fold0 = np.transpose(last_outputs_fold0, (2, 1, 0)) # z y x
    # last_outputs_fold1 = np.transpose(last_outputs_fold1, (2, 1, 0))
    last_outputs_fold2 = np.transpose(last_outputs_fold2, (2, 1, 0))
    last_outputs_fold3 = np.transpose(last_outputs_fold3, (2, 1, 0))
    last_outputs_fold4 = np.transpose(last_outputs_fold4, (2, 1, 0))
    last_outputs_fold5 = np.transpose(last_outputs_fold5, (2, 1, 0))
    last_outputs_fold0 = last_outputs_fold0 + last_outputs_fold2 + last_outputs_fold3 + last_outputs_fold4 + last_outputs_fold5
    last_outputs_fold0 = np.where(last_outputs_fold0 >= 3, 1, 0)
    last_outputs_sitk = convert_to_sitk(last_outputs_fold0, new_spacing, ct_sitk.GetOrigin()) # x y z
    last_outputs_sitk = resample(last_outputs_sitk, target_spacing, recon_size)
    last_outputs = np.transpose(sitk.GetArrayFromImage(last_outputs_sitk), (2, 1, 0)) # x y z
    last_outputs[last_outputs < 0.5] = 0
    last_outputs[last_outputs >=0.5] = 1
    crop_metadata_arr = np.array(ary_crop_meta, dtype=np.uint16)
    crop_metadata_arr_round = crop_metadata_arr.astype("uint16")
    
    posttime = time.time()
    print(posttime - secpptime)
    bs = crop_metadata_arr_round[0]
    be = crop_metadata_arr_round[1]
    
    #before transpose
    z_len = round(nii_im_arr.shape[2])
    
    # 이미지 사이즈가 32 이하일때 오류 발생. 검은색으로 filling해주는 과정 필요할듯
    # 근데 그런이미지 없을듯 해서 일단은 그대로
    
    if meta_size[0] <= 512:
        prd_la_arr_3d = np.pad(last_outputs, ((bs[0]//2, 512-be[0]//2), (bs[1]//2, 512-be[1]//2), (bs[2], z_len-be[2])))
    else:
        prd_la_arr_3d = np.pad(last_outputs, ((bs[0], 1024-be[0]), (bs[1], 1024-be[1]), (bs[2], z_len-be[2])))
    res_prd_la_arr = prd_la_arr_3d.astype('uint8')
    res_prd_la_arr = np.transpose(res_prd_la_arr, (2, 1, 0)) # z y x
    prediction_label_sitk = sitk.GetImageFromArray(res_prd_la_arr)
    if half_mode == 0:
        prediction_label_sitk.SetDirection(ct_sitk.GetDirection())
        prediction_label_sitk.SetSpacing(ct_sitk.GetSpacing())
        prediction_label_sitk.SetOrigin(ct_sitk.GetOrigin())
    elif half_mode == 1:
        prediction_label_sitk.SetDirection(meta_direction)
        prediction_label_sitk.SetSpacing(meta_spacing)
        prediction_label_sitk.SetOrigin(meta_origin)
    # sitk.WriteImage(prediction_label_sitk, os.path.join(r"D:\!HaN_Challenge\!ForArytenoid" ,"sibal.nii.gz"))
    pred_list.insert(0, (oar_label_dict_rev[0], prediction_label_sitk))

    return pred_list    

# oar_list = [ 
#                 "Arytenoid",
#                 "A_Carotid_L",
#                 "A_Carotid_R",
#                 "Bone_Mandible",
#                 "Brainstem",
#                 "BuccalMucosa",
#                 "Cavity_Oral",
#                 "Cochlea_L",
#                 "Cochlea_R",
#                 "Cricopharyngeus",
#                 "Esophagus_S",
#                 "Eye_AL",
#                 "Eye_AR",
#                 "Eye_PL",
#                 "Eye_PR",
#                 "Glnd_Lacrimal_L",
#                 "Glnd_Lacrimal_R",
#                 "Glnd_Submand_L",
#                 "Glnd_Submand_R",
#                 "Glnd_Thyroid",
#                 "Glottis",
#                 "Larynx_SG",
#                 "Lips",
#                 "OpticChiasm",
#                 "OpticNrv_L"
#                 "OpticNrv_R",
#                 "Parotid_L",
#                 "Parotid_R",
#                 "Pituitary",
#                 "SpinalCord",
#                 ]
        
# validation_basepath = r"D:\!HaN_Challenge\validation_fold0_cut"
# val_dir_list = os.listdir(validation_basepath)
# savepath =r"D:\!HaN_Challenge\validation_fold0_result_cut"
# os.makedirs(savepath, exist_ok=
#             True)
# for idx in range(len(val_dir_list)):
#     val_path = os.path.join(validation_basepath, val_dir_list[idx])
#     val_sitk = sitk.ReadImage(val_path)
#     pred_list = inference_hanseg(val_sitk)
#     for i, arr in enumerate(pred_list):
#         la_sitk = sitk.GetImageFromArray(arr)
#         sitk.WriteImage(la_sitk, os.path.join(savepath, "P%03d_%s.nii.gz" %(idx, oar_list[i])))
        