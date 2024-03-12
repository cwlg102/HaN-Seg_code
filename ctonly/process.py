import SimpleITK as sitk
import numpy as np
import logging

logger = logging.getLogger(__name__)

from custom_algorithm import Hanseg2023Algorithm
from inference_code_oar_nodet_skipver import inference_hanseg

LABEL_dict = {
    "background": 0,
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}
oar_label_dict = {'Arytenoid': 0, 'A_Carotid_L': 1, 'A_Carotid_R': 2, 'Bone_Mandible': 3, 'Brainstem': 4, 'BuccalMucosa': 5, 
           'Cavity_Oral': 6, 'Cochlea_L': 7, 'Cochlea_R': 8, 'Cricopharyngeus': 9, 'Esophagus_S': 10, 
           'Eye_AL': 11, 'Eye_AR': 12, 'Eye_PL': 13, 'Eye_PR': 14, 'Glnd_Lacrimal_L': 15, 
           'Glnd_Lacrimal_R': 16, 'Glnd_Submand_L': 17, 'Glnd_Submand_R': 18, 'Glnd_Thyroid': 19, 'Glottis': 20, 
           'Larynx_SG': 21, 'Lips': 22, 'OpticChiasm': 23, 'OpticNrv_L': 24, 'OpticNrv_R': 25, 
           'Parotid_L': 26, 'Parotid_R': 27, 'Pituitary': 28, 'SpinalCord': 29}

class MyHanseg2023Algorithm(Hanseg2023Algorithm):
    def __init__(self):
        super().__init__()

    def predict(self, *, image_ct: sitk.Image, image_mrt1: sitk.Image) -> sitk.Image:
        
        # create an empty segmentation same size as ct image
        # output_seg = image_ct * 0
        
        pred_list = inference_hanseg(image_ct)

        output_seg_arr = np.zeros_like(sitk.GetArrayFromImage(image_ct)).astype("uint8")
        for idx in range(len(pred_list)):
            oar_name, la_sitk = pred_list[idx]
            oar_number = LABEL_dict[oar_name]
            la_arr = sitk.GetArrayFromImage(la_sitk).astype("uint8")
            la_coord = np.where(la_arr == 1)
            output_seg_arr[la_coord[0], la_coord[1], la_coord[2]] = oar_number
        
        output_seg_sitk = sitk.GetImageFromArray(output_seg_arr)
        # inpaint a simple cuboid shape in the 3D segmentation mask
        # ct_shape = image_ct.GetSize()
        # output_seg[int(ct_shape[0]*0.1):int(ct_shape[0]*0.6), 
        #            int(ct_shape[1]*0.2):int(ct_shape[1]*0.7), 
        #            int(ct_shape[2]*0.3):int(ct_shape[2]*0.8)] = 1
        
        # output should be a sitk image with the same size, spacing, origin and direction as the original input image_ct
        output_seg_sitk.SetOrigin(image_ct.GetOrigin())
        output_seg_sitk.SetDirection(image_ct.GetDirection())
        output_seg_sitk.SetSpacing(image_ct.GetSpacing())

        output_seg_sitk = sitk.Cast(output_seg_sitk, sitk.sitkUInt8)
        return output_seg_sitk


if __name__ == "__main__":
    # import os
    # ct_sitk = sitk.ReadImage(r"D:\!HaN_Challenge\!TRAIN_FOLD_0\fold0_val_mha\P036_CT.mha")
    # mr_sitk = sitk.ReadImage(r"D:\!HaN_Challenge\!TRAIN_FOLD_0\fold0_val_mr_mha\case_36_IMG_MR_T1.mha")
    # MyHanseg2023Algorithm().predict(image_ct=ct_sitk, image_mrt1=mr_sitk)
    MyHanseg2023Algorithm().process()