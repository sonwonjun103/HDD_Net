import torch
import SimpleITK as sitk
import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, binary_erosion

class CustomDataset(Dataset):
    def __init__(self,
                 args,
                 ct,
                 hippo):
        super().__init__()

        self.args = args
        self.ct = ct
        self.hippo = hippo

    def __len__(self):
        return len(self.ct)
    
    # Get Volume
    def __get_volume(self, path, temp=0):
        volume = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(volume)

        if temp:
            volume = np.transpose(volume, (1, 0, 2))
            volume = np.rot90(volume, 2) 

        return volume
    
    # For CT volume function
    def __minmaxnormalize(self, volume):
        copy_volume = volume.copy()

        s = np.min(volume)
        b = np.max(volume)

        return (copy_volume - s) / (b - s)
    
    def __adjust__window(self, volume):
        copy_volume = volume.copy()

        min_window = self.args.min_window
        max_window = self.args.max_window

        copy_volume[copy_volume <= min_window] = min_window
        copy_volume[copy_volume >= max_window] = max_window

        return copy_volume
    
    def bone_processing(self, volume):
        copy_volume = volume.copy()

        copy_volume[copy_volume >= self.args.bone_threshold] = 0

        return copy_volume
    
    # For HIPPO volume function
    def __get_binary_volume(self, volume):
        copy_volume = volume.copy()

        copy_volume[copy_volume != 0] = 1
    
        return copy_volume 
    
    def __crop__volume(self, volume, crop_size):
        copy_volume = volume.copy() 

        d, h, w = volume.shape
        
        start_z = d // 2
        start_x = h // 2
        start_y = w // 2

        cropped_volume = copy_volume[start_z - crop_size[0] // 2 : start_z + crop_size[0] // 2,
                                    start_x - crop_size[1] // 2 : start_x + crop_size[1] // 2,
                                    start_y - crop_size[2] // 2 : start_y + crop_size[2] // 2,]
        
        return cropped_volume
    
    def get_boundary_map(self, volume):
        filter_data = gaussian_filter(volume, self.args.gaussian_filter)
        threshold = self.args.filter_threshold
 
        binary_mask = filter_data > threshold

        eroded_mask = binary_erosion(binary_mask)
        boundary_map = binary_mask.astype(int) - eroded_mask.astype(int)

        return boundary_map 
    
    def preprocessing(self, ct, hippo):

        ct = self.__crop__volume(ct, (self.args.depth_crop_size, self.args.crop_size, self.args.crop_size))
        hippo = self.__crop__volume(hippo, (self.args.depth_crop_size, self.args.crop_size, self.args.crop_size))
        
        ct = self.__adjust__window(ct)
        ct = self.__minmaxnormalize(ct)
        ct = self.bone_processing(ct)
        ct = self.__minmaxnormalize(ct)

        hippo = self.__get_binary_volume(hippo)
        boundary = self.get_boundary_map(hippo)

        boundary = self.__crop__volume(boundary, (self.args.depth_crop_size, self.args.crop_size, self.args.crop_size))


        return ct, hippo, boundary

    def __getitem__(self, index):
        ct_path = self.ct[index]
        hippo_path = self.hippo[index]
        
        ctvolume = self.__get_volume(ct_path, temp=1)
        hippovolume = self.__get_volume(hippo_path)

        """
        CT : Window size, Min Max normalization, Bone delete, Min Max, Crop
        HIPPO : Binary Image, Crop
        Boundary : Binary Image, Crop
        """

        ct, hippo, boundary = self.preprocessing(ctvolume, hippovolume)
        
        #sitk.WriteImage(sitk.GetImageFromArray(ct), './ct.nii.gz')
        #sitk.WriteImage(sitk.GetImageFromArray(hippo), './hippo.nii.gz')

        return torch.from_numpy(ct).unsqueeze(0), torch.from_numpy(hippo).unsqueeze(0), torch.from_numpy(boundary).unsqueeze(0)