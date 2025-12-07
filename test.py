import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from config import Config
# The opt file
opt = Config('./training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
# The testing dataset files
img_path = './dataset/CBSD68_bicubic_blur_jpeg_noise'
targeet_path = './dataset/CBSD68'
save_dir = './output_all_original'
img_list = sorted(os.listdir(img_path))
num_img = len(img_list)
import torch
torch.backends.cudnn.benchmark = True
import utils as utils
import os
from CR import ContrastLoss
from config import Config
from models.encoder2 import Convres
from restormer import Restormer_ours
pus = ','.join([str(i) for i in opt.GPU])
from torchvision.transforms import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import utils as utils
import time
import numpy as np
from model.common import VGGLoss
from data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from model_wave.mwdcnn import WMDCNN

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import lpips

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
contrast_loss = torch.nn.CrossEntropyLoss().cuda()
# device = torch.device("cuda:0,1")
start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
device_ids = [i for i in range(torch.cuda.device_count())]
######### Model ###########

model_G1 = WMDCNN()


######### Resume ###########
path_chk_rest = '/home/tione/notebook/home/lx/Low_light_rainy-main/checkpoint_all/Deraining/models/MPRNet/model_25.pth'
utils.load_checkpointG1(model_G1, path_chk_rest)
start_epoch = utils.load_start_epoch(path_chk_rest) + 1
print("start_epoch=",start_epoch)
# utils.load_optimG1(optimizer_G1, path_chk_rest)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
new_lr = opt.OPTIM.LR_INITIAL
loss_vgg = VGGLoss().cuda()
optimizer_G1 = optim.Adam(model_G1.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
######### Scheduler ###########
warmup_epochs = 0


if len(device_ids) > 1:
    model_G1 = nn.DataParallel(model_G1, device_ids=device_ids)

model_G1.cuda()
######### Loss ###########
ide_loss = torch.nn.L1Loss().cuda()
satu = ContrastLoss().cuda()
######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False, pin_memory=True)


print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
transform = transforms.ToTensor()
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    print("laileao")
    model_G1.eval()
    transform = transforms.ToTensor()
    lpips_alex = lpips.LPIPS(net='alex').cuda()
    PSNR = 0
    SSIM = 0
    MSE = 0
    LPIPS_score = 0

    # testing stage
    for img in img_list:
        image = Image.open(img_path + '/' + img).convert('RGB')
        name, _ = os.path.splitext(img)
        tarpath = os.path.join(targeet_path, name + ".bmp")

        target = Image.open(tarpath).convert('RGB')
        image = transform(image).unsqueeze(0).cuda()   # [1, 3, H, W]
        target = transform(target).unsqueeze(0).cuda() # [1, 3, H, W]
        with torch.set_grad_enabled(False):
            pre= model_G1(image)

        p_numpy = pre.squeeze(0).cpu().detach().numpy()
        label_numpy = target.squeeze(0).cpu().detach().numpy()

        psnr = peak_signal_noise_ratio(label_numpy, p_numpy, data_range=1)
        PSNR += psnr

        out_img = pre.squeeze(0).cpu().detach().clamp(0,1)  # [3,H,W], 限制到[0,1]
        out_img = transforms.ToPILImage()(out_img)          # 转 PIL
        out_img.save(os.path.join(save_dir, name + "_restored.png"))

        p_img = np.transpose(p_numpy, (1,2,0))
        label_img = np.transpose(label_numpy, (1,2,0))
        ssim = structural_similarity(label_img, p_img, channel_axis=2, data_range=1)
        SSIM += ssim

        # MSE
        mse = mean_squared_error(label_img, p_img)
        MSE += mse

        # LPIPS (输入需为 [-1,1] tensor)
        pre_norm = (pre * 2) - 1
        target_norm = (target * 2) - 1
        lpips_val = lpips_alex(pre_norm, target_norm)
        LPIPS_score += lpips_val.item()

PSNR /= num_img
SSIM /= num_img
MSE /= num_img
LPIPS_score /= num_img

print(f"PSNR  = {PSNR:.4f}")
print(f"SSIM  = {SSIM:.4f}")
print(f"MSE   = {MSE:.6f}")
print(f"LPIPS = {LPIPS_score:.4f}")
print(f"[Info] 保存的结果图像在 {save_dir}")