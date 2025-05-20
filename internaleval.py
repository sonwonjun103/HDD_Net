import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk

from models.moduleRCA import Model
from utils.utils import dice_coefficient, iou_coefficient
from options.TestOption import TestParser
from scipy.ndimage import gaussian_filter, binary_erosion
from sklearn.metrics import confusion_matrix

class Eval():
    def __init__(self,
                 args,
                 test_ct,
                 test_hippo,  
                 model):
        self.args = args
        self.model = model
        self.model_save_path = os.path.join("./model_parameters", f"{args.model}_{args.filename}.pt")
        self.test_ct = test_ct
        self.test_hippo = test_hippo
        self.save_path = f"D:\\HIPPO\\{args.date}\\test"

    def get_eval_metric(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()

        cm = confusion_matrix(pred, target)
        TP = cm[1][1]
        TN = cm[0][0]

        FP = cm[0][1]
        FN = cm[1][0]

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = TP / (TP + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        return f1_score, iou, precision, recall, accuracy

    # def remove_false_negative(self, target, pred):
    #     d, h, w = target.shape

    #     for i in range(d):
    #         for j in range(h):
    #             for z in range(w):
    #                 # target에서 0

    def load(self):
        model = self.model.to(self.args.device)
        model = torch.nn.DataParallel(model).to(self.args.device)
        self.model.load_state_dict(torch.load(self.model_save_path))
        print(f"Model Load Complete!")

    def get_volume(self, path, temp=0):
        volume = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(volume)

        if temp:
            volume = np.transpose(volume, (1,0,2))
            volume = np.rot90(volume, 2) 

        return volume
    
    def thresholding(self, volume, threshold):
        copy_volume = volume.copy()

        copy_volume[copy_volume > threshold] = 1
        copy_volume[copy_volume <= threshold] = 0

        return copy_volume

    def crop__volume(self, volume, crop_size):
        copy_volume = volume.copy()

        d, h, w = volume.shape
        
        start_z = d // 2
        start_x = h // 2
        start_y = w // 2

        cropped_volume = copy_volume[start_z - crop_size[0] // 2 : start_z + crop_size[0] // 2,
                                    start_x - crop_size[1] // 2 : start_x + crop_size[1] // 2,
                                    start_y - crop_size[2] // 2 : start_y + crop_size[2] // 2,]
        
        return cropped_volume

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
    
    def get_boundary_map(self, volume):
        filter_data = gaussian_filter(volume, self.args.gaussian_filter)
        threshold = self.args.filter_threshold

        binary_mask = filter_data > threshold

        eroded_mask = binary_erosion(binary_mask)
        boundary_map = binary_mask.astype(int) - eroded_mask.astype(int)

        return boundary_map

    def make_data(self, index):
        test_ct = self.test_ct[index]
        test_hippo = self.test_hippo[index]

        # load data
        ct = self.get_volume(test_ct, temp=1)
        hippo = self.get_volume(test_hippo)

        # crop volume
        ct = self.crop__volume(ct, (self.args.depth_crop_size, 
                                    self.args.crop_size,
                                    self.args.crop_size))
        hippo = self.crop__volume(hippo, (self.args.depth_crop_size, 
                                    self.args.crop_size,
                                    self.args.crop_size))
        
        ct = self.__adjust__window(ct)
        ct = self.__minmaxnormalize(ct)
        ct = self.bone_processing(ct)
        ct = self.__minmaxnormalize(ct)

        hippo = self.__get_binary_volume(hippo)
        boundary = self.get_boundary_map(hippo)

        boundary = self.crop__volume(boundary, (self.args.depth_crop_size, self.args.crop_size, self.args.crop_size))

        return torch.from_numpy(ct).unsqueeze(0).unsqueeze(0), torch.from_numpy(hippo).unsqueeze(0).unsqueeze(0), torch.from_numpy(boundary).unsqueeze(0).unsqueeze(0)
    
    def gpu_to_cpu(self, volume, temp=None):
        if temp:
            return volume.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            return volume.squeeze(0).detach().cpu().numpy()
        
    # self.save_volume(ct, folder, model_folder, 'ct')
    def save_volume(self, volume, folder, model_folder, name, feature_folder=None):
        length = len(volume.shape)
        if length == 3:
            sitk.WriteImage(sitk.GetImageFromArray(volume), 
                            f"./internal_resultnii\\{folder}\\{model_folder}\\{name}.nii.gz")

        elif length == 4:
            slice = volume.shape[0]
            path = f"./internal_resultnii\\{folder}\\{model_folder}\\{feature_folder}"
            os.makedirs(f"{path}", exist_ok=True)

            for i in range(slice):
                sitk.WriteImage(sitk.GetImageFromArray(volume[i]),
                                os.path.join(path, f"{i+1}th_feature_map.nii.gz"))

    def evaluation(self):
        # load model
        self.load()

        # make data
        datasize = len(self.test_ct)
        print(f"test size : {datasize}")

        total_dice0, total_iou0, total_pre0, total_recall0, total_acc0 = [],[],[],[],[]
        total_dice1, total_iou1, total_pre1, total_recall1, total_acc1 = [],[],[],[],[]
        total_dice2, total_iou2, total_pre2, total_recall2, total_acc2 = [],[],[],[],[]
        total_dice3, total_iou3, total_pre3, total_recall3, total_acc3 = [],[],[],[],[]
        total_dice4, total_iou4, total_pre4, total_recall4, total_acc4 = [],[],[],[],[]
        total_dice5, total_iou5, total_pre5, total_recall5, total_acc5 = [],[],[],[],[]
        total_dice6, total_iou6, total_pre6, total_recall6, total_acc6 = [],[],[],[],[]
        total_dice7, total_iou7, total_pre7, total_recall7, total_acc7 = [],[],[],[],[]
        total_dice8, total_iou8, total_pre8, total_recall8, total_acc8 = [],[],[],[],[]
        total_dice9, total_iou9, total_pre9, total_recall9, total_acc9 = [],[],[],[],[]

        for i in range(datasize):
            ct, hippo, edge = self.make_data(i)
            folder = self.test_ct[i].split('\\')[3]

            model_folder = f"{self.args.model}_{self.args.filename}"     
            
            if os.path.exists(f"D:\\HIPPO\\FINALCODE\\internal_resultnii\\{folder}\\{model_folder}") == 1:
                print(f"{i+1} {folder} exists!")
                continue                                                            

            os.makedirs(f"D:\\HIPPO\\FINALCODE\\internal_resultnii\\{folder}", exist_ok=True)
            os.makedirs(f"D:\\HIPPO\\FINALCODE\\internal_resultnii\\{folder}\\{model_folder}", exist_ok=True)
            
            # predict
            if self.args.edge == 1:
                if self.args.module:
                    pred, edge_pred = self.model(ct.to(self.args.device).float())
                    
                else:
                    pred, edge_pred, \
                    x1, x2, x3, x4, x5, x1_d, x2_d, x3_d, x4_d, \
                    edge_x1_d, edge_x2_d, edge_x3_d, edge_x4_d = self.model(ct.to(self.args.device).float())

            else:
                pred = self.model(ct.to(self.args.device).float())
            
            # ct, hippo, edge => cpu && save
            ct = self.gpu_to_cpu(ct, 1)
            hippo = self.gpu_to_cpu(hippo, 1)
            edge = self.gpu_to_cpu(edge, 1)
            
            self.save_volume(ct, folder, model_folder, 'ct')
            self.save_volume(hippo, folder, model_folder, 'hippo')            
            self.save_volume(edge, folder, model_folder, 'edge')
            
            # save original pred
            threshold = 0.4
            pred = np.clip(self.gpu_to_cpu(pred, 1), 0, 1)
            self.save_volume(self.thresholding(pred, threshold), folder, model_folder, 'pred')

            if self.args.edge == 1:
                edge_pred = self.thresholding(self.gpu_to_cpu(edge_pred, 1), threshold)
                self.save_volume(edge_pred, folder, model_folder, 'edge_pred____')
                print(f"Edge saved!")
            print(f"{i+1} folder {folder} save!")

            # save feature map
            ## encoder feature map
            #if self.args.save == 1:
                # self.save_volume(self.gpu_to_cpu(x1), folder, model_folder, '', 
                #                 feature_folder='encoder1')
                # self.save_volume(self.gpu_to_cpu(x2), folder, model_folder, '', 
                #                 feature_folder='encoder2')
                # self.save_volume(self.gpu_to_cpu(x3), folder, model_folder, '', 
                #                 feature_folder='encoder3')
                # self.save_volume(self.gpu_to_cpu(x4), folder, model_folder, '', 
                #                 feature_folder='encoder4')
                # self.save_volume(self.gpu_to_cpu(x5), folder, model_folder, '', 
                #                 feature_folder='encoder5')
                
                # ## seg decoder feature map
                # self.save_volume(self.gpu_to_cpu(x1_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder1')
                # self.save_volume(self.gpu_to_cpu(x2_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder2')
                # self.save_volume(self.gpu_to_cpu(x3_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder3')
                # self.save_volume(self.gpu_to_cpu(x4_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder4')
    
                # ## edge decoder feature map
                # if self.args.edge:
                #     self.save_volume(self.gpu_to_cpu(edge_x1_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder1')
                #     self.save_volume(self.gpu_to_cpu(edge_x2_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder2')
                #     self.save_volume(self.gpu_to_cpu(edge_x3_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder3')
                #     self.save_volume(self.gpu_to_cpu(edge_x4_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder4')
    
                # ## Module feature map
                # if self.args.module:
                #     self.save_volume(self.gpu_to_cpu(module1_seg), folder, model_folder, '', 
                #                     feature_folder='module1_seg')
                #     self.save_volume(self.gpu_to_cpu(module1_edge), folder, model_folder, '', 
                #                     feature_folder='module1_edge')
                #     self.save_volume(self.gpu_to_cpu(module2_seg), folder, model_folder, '', 
                #                     feature_folder='module2_seg')
                #     self.save_volume(self.gpu_to_cpu(module2_edge), folder, model_folder, '', 
                #                     feature_folder='module2_edge')

            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                dice, iou, pre, recall, acc = self.get_eval_metric(self.thresholding(pred, threshold), hippo)

                if threshold == 0:
                    total_dice0.append(dice)
                    total_iou0.append(iou)
                    total_pre0.append(pre)
                    total_recall0.append(recall)
                    total_acc0.append(acc)
                elif threshold == 0.1:
                    total_dice1.append(dice)
                    total_iou1.append(iou)
                    total_pre1.append(pre)
                    total_recall1.append(recall)
                    total_acc1.append(acc)
                elif threshold == 0.2:
                    total_dice2.append(dice)
                    total_iou2.append(iou)
                    total_pre2.append(pre)
                    total_recall2.append(recall)
                    total_acc2.append(acc)
                elif threshold == 0.3:
                    total_dice3.append(dice)
                    total_iou3.append(iou) 
                    total_pre3.append(pre)
                    total_recall3.append(recall)
                    total_acc3.append(acc)
                elif threshold == 0.4:
                    total_dice4.append(dice)
                    total_iou4.append(iou) 
                    total_pre4.append(pre)
                    total_recall4.append(recall)
                    total_acc4.append(acc)
                elif threshold == 0.5:
                    total_dice5.append(dice)
                    total_iou5.append(iou) 
                    total_pre5.append(pre)
                    total_recall5.append(recall)
                    total_acc5.append(acc)
                elif threshold == 0.6:
                    total_dice6.append(dice)
                    total_iou6.append(iou)
                    total_pre6.append(pre)
                    total_recall6.append(recall)
                    total_acc6.append(acc)
                elif threshold == 0.7:
                    total_dice7.append(dice)
                    total_iou7.append(iou)
                    total_pre7.append(pre)
                    total_recall7.append(recall)
                    total_acc7.append(acc)
                elif threshold == 0.8:
                    total_dice8.append(dice)
                    total_iou8.append(iou)
                    total_pre8.append(pre)
                    total_recall8.append(recall)
                    total_acc8.append(acc)
                elif threshold == 0.9:
                    total_dice9.append(dice)
                    total_iou9.append(iou) 
                    total_pre9.append(pre)
                    total_recall9.append(recall)
                    total_acc9.append(acc)
                        
                print(f"{folder} {threshold}=>  Dice : {dice:>.3f} IOU : {iou:>.3f} Pre : {pre:>.3f} Recall : {recall:>.3f} ACC : {acc:>.3f}")
                #torch.cuda.empty_cache()
            print()
          
        print(f"Mean Dice 0.0 : {np.mean(total_dice0):>.3f} {np.std(total_dice0):>.3f}")
        print(f"Mean IOU  0.0 : {np.mean(total_iou0):>.3f} {np.std(total_iou0):>.3f}")
        print(f"Mean Precision 0.0 : {np.mean(total_pre0):>.3f} {np.std(total_pre0):>.3f}")
        print(f"Mean Recall 0.0 : {np.mean(total_recall0):>.3f} {np.std(total_recall0):>.3f}")
        print(f"Mean ACC 0.0 : {np.mean(total_acc0):>.3f} {np.std(total_acc0):>.3f}")
        print()
        print(f"Mean Dice 0.1 : {np.mean(total_dice1):>.3f} {np.std(total_dice1):>.3f}")
        print(f"Mean IOU  0.1 : {np.mean(total_iou1):>.3f} {np.std(total_iou1):>.3f}")
        print(f"Mean Precision 0.1 : {np.mean(total_pre1):>.3f} {np.std(total_pre1):>.3f}")
        print(f"Mean Recall 0.1 : {np.mean(total_recall1):>.3f} {np.std(total_recall1):>.3f}")
        print(f"Mean ACC 0.1 : {np.mean(total_acc1):>.3f} {np.std(total_acc1):>.3f}")
        print()
        print(f"Mean Dice 0.2 : {np.mean(total_dice2):>.3f} {np.std(total_dice2):>.3f}")
        print(f"Mean IOU  0.2 : {np.mean(total_iou2):>.3f} {np.std(total_iou2):>.3f}")
        print(f"Mean Precision 0.2 : {np.mean(total_pre2):>.3f} {np.std(total_pre2):>.3f}")
        print(f"Mean Recall 0.2 : {np.mean(total_recall2):>.3f} {np.std(total_recall2):>.3f}")
        print(f"Mean ACC 0.2 : {np.mean(total_acc2):>.3f} {np.std(total_acc2):>.3f}")
        print()
        print(f"Mean Dice 0.3 : {np.mean(total_dice3):>.3f} {np.std(total_dice3):>.3f}")
        print(f"Mean IOU  0.3 : {np.mean(total_iou3):>.3f} {np.std(total_iou3):>.3f}")
        print(f"Mean Precision 0.3 : {np.mean(total_pre3):>.3f} {np.std(total_pre3):>.3f}")
        print(f"Mean Recall 0.3 : {np.mean(total_recall3):>.3f} {np.std(total_recall3):>.3f}")
        print(f"Mean ACC 0.3 : {np.mean(total_acc3):>.3f} {np.std(total_acc3):>.3f}")
        print()
        print(f"Mean Dice 0.4 : {np.mean(total_dice4):>.3f} {np.std(total_dice4):>.3f}")
        print(f"Mean IOU  0.4 : {np.mean(total_iou4):>.3f} {np.std(total_iou4):>.3f}")
        print(f"Mean Precision 0.4 : {np.mean(total_pre4):>.3f} {np.std(total_pre4):>.3f}")
        print(f"Mean Recall 0.4 : {np.mean(total_recall4):>.3f} {np.std(total_recall4):>.3f}")
        print(f"Mean ACC 0.4 : {np.mean(total_acc4):>.3f} {np.std(total_acc4):>.3f}")
        print()
        print(f"Mean Dice 0.5 : {np.mean(total_dice5):>.3f} {np.std(total_dice5):>.3f}")
        print(f"Mean IOU  0.5 : {np.mean(total_iou5):>.3f} {np.std(total_iou5):>.3f}")
        print(f"Mean Precision 0.5 : {np.mean(total_pre5):>.3f} {np.std(total_pre5):>.3f}")
        print(f"Mean Recall 0.5 : {np.mean(total_recall5):>.3f} {np.std(total_recall5):>.3f}")
        print(f"Mean ACC 0.5 : {np.mean(total_acc5):>.3f} {np.std(total_acc5):>.3f}")
        print()
        print(f"Mean Dice 0.6 : {np.mean(total_dice6):>.3f} {np.std(total_dice6):>.3f}")
        print(f"Mean IOU  0.6 : {np.mean(total_iou6):>.3f} {np.std(total_iou6):>.3f}")
        print(f"Mean Precision 0.6 : {np.mean(total_pre6):>.3f} {np.std(total_pre6):>.3f}")
        print(f"Mean Recall 0.6 : {np.mean(total_recall6):>.3f} {np.std(total_recall6):>.3f}")
        print(f"Mean ACC 0.6 : {np.mean(total_acc6):>.3f} {np.std(total_acc6):>.3f}")
        print()
        print(f"Mean Dice 0.7 : {np.mean(total_dice7):>.3f} {np.std(total_dice7):>.3f}")
        print(f"Mean IOU  0.7 : {np.mean(total_iou7):>.3f} {np.std(total_iou7):>.3f}")
        print(f"Mean Precision 0.7 : {np.mean(total_pre7):>.3f} {np.std(total_pre7):>.3f}")
        print(f"Mean Recall 0.7 : {np.mean(total_recall7):>.3f} {np.std(total_recall7):>.3f}")
        print(f"Mean ACC 0.7 : {np.mean(total_acc7):>.3f} {np.std(total_acc7):>.3f}")
        print()
        print(f"Mean Dice 0.8 : {np.mean(total_dice8):>.3f} {np.std(total_dice8):>.3f}")
        print(f"Mean IOU  0.8 : {np.mean(total_iou8):>.3f} {np.std(total_iou8):>.3f}")
        print(f"Mean Precision 0.8 : {np.mean(total_pre8):>.3f} {np.std(total_pre8):>.3f}")
        print(f"Mean Recall 0.8 : {np.mean(total_recall8):>.3f} {np.std(total_recall8):>.3f}")
        print(f"Mean ACC 0.8 : {np.mean(total_acc8):>.3f} {np.std(total_acc8):>.3f}")
        print()
        print(f"Mean Dice 0.9 : {np.mean(total_dice9):>.3f} {np.std(total_dice9):>.3f}")
        print(f"Mean IOU  0.9 : {np.mean(total_iou9):>.3f} {np.std(total_iou9):>.3f}")
        print(f"Mean Precision 0.9 : {np.mean(total_pre9):>.3f} {np.std(total_pre9):>.3f}")
        print(f"Mean Recall 0.9 : {np.mean(total_recall9):>.3f} {np.std(total_recall9):>.3f}")
        print(f"Mean ACC 0.9 : {np.mean(total_acc9):>.3f} {np.std(total_acc9):>.3f}")
        print()



if __name__=='__main__':
    opt = TestParser()
    args = opt.parse()

    device = args.device
    print(f"Device : {device}")

    test_ct = pd.read_excel(f"D:\\HIPPO\\test_plus.xlsx")['CT']
    test_hippo = pd.read_excel(f"D:\\HIPPO\\test_plus.xlsx")['HIPPO']
    #model = AGUnet(1,1).to(device)
    model = Model(1, 1).to(device)
    #model = Unetedge(1, 1).to(device)
    # model = SSLHead().to(device)
    # model = UNETR(
    #     in_channels=1,
    #     out_channels=1,
    #     img_size=(96, 128, 128),
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     norm_name='instance',
    #     res_block=True,
    #     conv_block=True,
    #     dropout_rate=0.0
    # )
    # model = Model(
    #     in_channels=1,
    #     out_channels=1,
    #     final_sigmoid=False
    # )
    model = torch.nn.DataParallel(model).to(device)

    evaluator = Eval(args,
                     test_ct,
                     test_hippo,
                     model)
    
    evaluator.evaluation()