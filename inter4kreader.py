import numpy as np
import os,sys,cv2,torch
import torch.nn.functional as F
import torch.utils.data as data
# Datareader for inter4K testset.
class Inter4K_Test(data.Dataset):
    def __init__(self, args, scenerange=8):
        self.args = args

        # Since we do 8x interpolation, multiple is always set to 8
        self.multiple = 8
        self.scenerange = scenerange
        # check if scenerange number makes sense with given multiple number
        assert self.scenerange%self.multiple == 0

        self.testPath = []
        testfolder = self.args.inter4k_data_path
        folders = os.listdir(testfolder)
        t = np.linspace((1 / self.multiple), (1 - (1 / self.multiple)), (self.multiple - 1))
        
        # Generate a big list containing every test datapoint. 
        # Each datapoint consists out of two input frames and one ground truth middle frame.
        for i in folders:
            tempath = os.path.join(testfolder,i)
            

            temfiles = sorted(os.listdir(tempath),key=lambda x: int(x.split("_")[0][3:]))
            scenes = [[]]
            lastscene = 0
            for imadd in temfiles:
                if(int(imadd.split("_")[1][:-4])> lastscene):
                    scenes.append([imadd])
                    lastscene += 1
                else:
                    scenes[-1].append(imadd)

            # with scenerange=16, every second frame is skipped as ground truth
            # one folder can contain multiple scenes!
            for scenindex,scen in enumerate(scenes):
                if(not len(scen)< self.scenerange+1):
                    temscen = [os.path.join(tempath,file) for file in scen]
                    for temk in range(self.multiple-1):
                        self.testPath.append([temscen[0],temscen[self.scenerange],temscen[(temk+1)*(self.scenerange//self.multiple)],t[temk],tempath+"_scene_"+str(scenindex)])

   
            

        print("# of Inter4K triplet testset : ",len(self.testPath))
        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.test_data_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]

        # Loading the actual frames and normalizing them.
        frames = frames_loader_test(self.args, I0I1It_Path,validation=False)


        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


def frames_loader_test(args, I0I1It_Path, validation,Xiph=False):
    frames = []
    for path in I0I1It_Path:

        if(Xiph):
            frame = cv2.imread(path,flags=-1)
        else:
            frame = cv2.imread(path)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)


    if args.dataset == 'X4K1000FPS':
        if validation:
            ps = args.validation_patch_size#512
        else:
            ps = args.test_patch_size
        if(ps != -1):
            ix = (iw - ps) // 2
            iy = (ih - ps) // 2
            frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames

def RGBframes_np2Tensor(imgIn, channel):

    if channel == 1:

        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (channel, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn