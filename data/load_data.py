import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split

def get_path(args):
    path = f"D:\\HIPPO\\DATA\\"
    f=[]

    for folder in os.listdir(path):
        f.append(folder)

    train_ct = []
    train_hippo = []

    test_ct = []
    test_hippo = []

    test = ['83630','340165','400513','507579', '515629', '579709',
            '744583','752222','774490','1368064','1373675','1381826','1492672','1510895',
            '1888224','1937205','2620964','2806577','2832554', 
            '3048080','3188716','3451470','3757646','3792010','5083081','9448675','9453941','9457681','9458092','9440878',
            ] 

    w = ['39875','810960','947467','1225436','1418261','2115879','3289530',
         '3772956','3939496','4110906','5015004','5590382','6144646','6147846','6344793','6432096','6477005',
         '489357', '531159','893437']

    for train_f in f:
        if train_f in w or train_f in test:
            continue
        folder_path = f"{path}{train_f}"
        for file in glob.glob(folder_path + "/*.nii"):
            filename = file.split('\\')[4]
            if filename.startswith('r') and filename.endswith('_CT.nii'):
                train_ct.append(file)
            elif filename.startswith('hippovolume'):
                train_hippo.append(file)
            # elif filename.startswith('boundary'):
            #     train_edge.append(file)

    for test_f in test:
        folder_path = f"{path}{test_f}"
        for file in glob.glob(folder_path + "/*.nii"):
            filename = file.split('\\')[4]
            if filename.startswith('r') and filename.endswith('_CT.nii'):
                test_ct.append(file)
            elif filename.startswith('hippovolume'):
                test_hippo.append(file)
            # elif filename.startswith('edge'):
            #     test_edge.append(file)
                
    print(f"Train CT : {len(train_ct)}, Test CT : {len(test_ct)}")
    print(f"Train HIPPO : {len(train_hippo)}, Test HIPPO : {len(test_hippo)}")
    #print(len(train_edge), len(test_edge))

    train_frame = pd.DataFrame({'CT': train_ct,
                                'HIPPO' : train_hippo})
    
    test_frame = pd.DataFrame({'CT' : test_ct,
                               'HIPPO' : test_hippo})
    
    train_frame.to_excel(f"D:\\HIPPO\\train_.xlsx", index=False)
    test_frame.to_excel(f"D:\\HIPPO\\test_.xlsx", index=False)