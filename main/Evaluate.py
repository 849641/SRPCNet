import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model.get_model import get_model

def Eval(Epoch, Dataset):

    print('Begin Eval \n -----------------------------------------')

    # 加载当前epoch权重
    weights_path = os.path.join('../checkpoint/' + 'SRPC' + str(Epoch))
    print('Eval with :' + 'SRPC' + str(Epoch))

    # eval_dataset
    eval_path = '../SSP2000/Test'
    cfg = Dataset.Config(datapath=eval_path, snapshot=weights_path, mode='eval')
    data = Dataset.Data(cfg, 'SRPC')
    eval_dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)

    # network
    net = get_model(cfg, 'SRPC')
    net.train(False)
    net.cuda()

    for image, (H, W), name in tqdm(eval_dataloader):

        with torch.no_grad():
            image, shape = image.cuda().float(), (H, W)
            out1, out2, out3, out4, out5 = net(image, shape, name)
            pred = torch.sigmoid(out1[0, 0]).cpu().numpy() * 255
            savepath = os.path.join('../Prediction/SRPC/' + str(Epoch))
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            cv2.imwrite(savepath+'/'+name[0]+'.png', np.round(pred))

    print('Done with :' + 'SRPC' + str(Epoch))
