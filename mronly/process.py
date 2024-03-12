import SimpleITK as sitk
import numpy as np
import logging
import matplotlib.pyplot as plt 

logger = logging.getLogger(__name__)

from custom_algorithm import Hanseg2023Algorithm
from inference_code_oar_nodet_skipver import inference_hanseg

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

def HistogramNorm(mr_volume):
    mr_one_dim = np.ravel(mr_volume)
    print(np.max(mr_volume), np.min(mr_volume))
    if np.min(mr_volume) < 0:
        mr_volume += abs(np.min(mr_volume))
    
    mr_volume -= np.min(mr_volume)
    #####normalization#####
    max_pixel_value = 3000
    mrnorm_hyperparam = 1000
    counts_list, bin_locations, patches = plt.hist(mr_one_dim, max_pixel_value, (0, max_pixel_value))
    plt.ylim((0, 1e5))
    plt.xlim((0, 2500))
    plt.xlabel("Pixel Value")
    plt.ylabel("Number of Pixels")
    # plt.show()
    
    for idx_val in range(max_pixel_value-1, -1, -1):
        if counts_list[idx_val] > mrnorm_hyperparam:
            val_norm = idx_val+1
            break
        
    mr_volume = np.where(mr_volume > val_norm, val_norm, mr_volume)
    return mr_volume
def registration_code(fixed_sitk, moving_sitk, min_val, param_save_path = None, ind = None, temp_idx = 0):
    """1) Set ElastixImageFilter"""
    elastixImageFilter = sitk.ElastixImageFilter()
    """2) Set Parameters"""
    elastixImageFilter.SetFixedImage(fixed_sitk)
    elastixImageFilter.SetMovingImage(moving_sitk)
    # elastixImageFilter.SetFixedMask(fixed_mask_sitk)
    
    parameterMapVector = sitk.VectorOfParameterMap()
    translation_map = sitk.GetDefaultParameterMap('translation')
    translation_map['MaximumNumberOfIterations'] = ['200']
    translation_map['DefaultPixelValue'] = [str(min_val)]
    translation_map["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    translation_map["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    translation_map["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
    # translation_map["ImagePyramidSchedule"] = ["8", "8", "4", "4", "2", "2", "1", "1"]
    translation_map["Interpolator"] = ["BSplineInterpolator"]
    # translation_map["Metric"] = ["AdvancedNormalizedCorrelation"]
    translation_map["Optimizer"] = ["StandardGradientDescent"]
    translation_map["SP_a"] = ["40000"]
    translation_map['Metric1Weight'] = ["1e-4"]
    # translation_map["Transform"] = ["TranslationTransform"]
    translation_map["AutomaticTransformInitialization"] = ["true"]
    translation_map["AutomaticScalesEstimation"] = ["true"]
    translation_map["FinalGridSpacingInPhysicalUnits"] = ["8.0", "4.0", "8.0"]
    translation_map["GridSpacingSchedule"] = ["8.0", "4.0", "2.5", "1.0"]
    translation_map["HowToCombineTransforms"] = ["Compose"]
    translation_map["AutomaticParameterEstimation"] = ["true"]
    translation_map["UseAdaptiveStepSizes"] = ["true"]
    translation_map["NumberOfHistogramBins"] = ["32"]
    translation_map["FixedKernelBSplineOrder"] = ["1"]
    translation_map["MovingKernelBSplineOrder"] = ["3"]
    translation_map["ImageSampler"] = ["RandomCoordinate"]
    # translation_map["NumberOfSpatialSamples"] = ["8192"]
    translation_map["NewSamplesEveryIteration"] = ["true"]

    parameterMapVector.append(translation_map)

    
    
    # parametermap['MaximumNumberOfSamplingAttempts'] = ['16']
    elastixImageFilter.SetParameterMap(parameterMapVector) #method
    # elastixImageFilter.SetInitialTransform(initial_transform, inPlace=True)
    """3) Execute"""
    elastixImageFilter.Execute()
    resultimage = elastixImageFilter.GetResultImage()
    
    #save parameter map
    
    # while True:
    #     try:
    #         sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[temp_idx], os.path.join(param_save_path, "P%03d_%03d_param.txt" %(ind, temp_idx)))
    #     except:
    #         break
    #     temp_idx += 1

    return resultimage
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
    
        mr_direction = np.array(image_mrt1.GetDirection())
        ct_direction = np.array(image_ct.GetDirection())
        change_direction = mr_direction * ct_direction
        

        mr_arr = sitk.GetArrayFromImage(image_mrt1)
        mr_arr = HistogramNorm(mr_arr.astype("int32"))
        mr_arr = np.int32(255 * (mr_arr.astype("float32") - np.min(mr_arr))/(np.max(mr_arr) - np.min(mr_arr)))

        image_mr = sitk.GetImageFromArray(mr_arr)
        image_mr.SetSpacing(image_mrt1.GetSpacing())
        image_mr.SetDirection(image_mrt1.GetDirection())
        image_mr.SetOrigin(image_mrt1.GetOrigin())
        image_mr = resample(image_mr, image_ct.GetSpacing(), image_ct.GetSize())
        image_mr = registration_code(image_ct, image_mr, 0)

        # print(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(im_sitk.GetDirection()))
        
        

        pred_list = inference_hanseg(image_ct, image_mr)

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