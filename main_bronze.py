import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.hub
import argparse
from model_bronze import AKG
from graph_loss import GraphLoss
from train_test_bronze import train
from bronze_dataset import BronzeWare_Dataset
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DATASET_ROOT = "./complete_DATASET"
Modelset = "./pretrained_resnet50"
exp_PATH = "./experiment_result"


seed = 2022
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True 
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--worker', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth', help='Path of pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--proportion', type=float, default=1.0, help='Proportion of species label')  
    parser.add_argument('--epoch', default=64, type=int, help='Epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')      
    parser.add_argument('--dataset', type=str, default='bronze', help='dataset name')
    parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual', choices=['Cos', 'Step'])
    parser.add_argument('--device', nargs='+', default='0', help='GPU IDs for DP training')
    parser.add_argument('--img_size', default=450, type=int, help='-')
    parser.add_argument('--input_size', default=400, type=int, help='-')
    parser.add_argument('--BATCH_SIZE', default=32, type=int, help='-')
    parser.add_argument('--lr', default=1e-4, type=float, help='-')
    parser.add_argument('--alph1', default=2, type=int)
    parser.add_argument('--alph2', default=3, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--Lambda', default=0.1, type=float)
    parser.add_argument('--exp_name', default="Bronze_Dating_EXP", type=str)
    parser.add_argument('--exp_path', default=exp_PATH, type=str)
    parser.add_argument('--sig_threshold', default=0.8, type=float)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    for EXP_number in range(1,2):
        args = arg_parse()
        exp_ROOT = os.path.join(exp_PATH, args.exp_name)
        if os.path.exists(exp_ROOT) is False:
            os.makedirs(exp_ROOT)
        if os.path.exists(os.path.join(exp_ROOT, "bronze_pt")) is not True:
            os.mkdir(os.path.join(exp_ROOT, "bronze_pt"))

        print("parameter setting: alph1 = %.5f,alph2 = %.5f,beta = %.5f, lambda = %.6f" % (args.alph1, args.alph2, args.beta, args.Lambda))
        print('==> proportion: ', args.proportion)
        print('==> epoch: ', args.epoch)
        print('==> batch: ', args.BATCH_SIZE)
        print('==> dataset: ', args.dataset)
        print('==> img_size: ', args.img_size)
        print('==> device: ', args.device)
        print('==> Schedual: ', args.lr_adjt)

        nb_epoch = args.epoch
        batch_size = args.batch
        num_workers = args.worker

        transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomCrop(args.input_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        if args.dataset == 'bronze':
            levels = 2
            total_nodes = 15
            trees = [
                [4,0],
                [5,0],
                [6,1],
                [7,1],
                [8,1],
                [9,2],
                [10,2],
                [11,2],
                [12,3],
                [13,3],
                [14,3]
            ]
        
        data_path = DATASET_ROOT+"/delete_png"
        xml_path = DATASET_ROOT+"/xml_all"
        train_excel_path = DATASET_ROOT+"/train.xlsx"
        val_excel_path = DATASET_ROOT+"/val.xlsx"
        test_excel_path = DATASET_ROOT+"/test.xlsx"

        trainset = BronzeWare_Dataset(data_path, xml_path, train_excel_path, transform_train, train=True, size=args.input_size)
        valset = BronzeWare_Dataset(data_path, xml_path, val_excel_path, transform_test, train=False, size=args.input_size)
        testset = BronzeWare_Dataset(data_path, xml_path, test_excel_path, transform_test, train=False, size=args.input_size)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True, collate_fn=trainset.collate_fn)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = True, collate_fn=valset.collate_fn)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = True, collate_fn=testset.collate_fn)

        # GPU
        device = torch.device("cuda:" + args.device[0])
        
        backbone = models.resnet50(pretrained=False)
        backbone.load_state_dict(torch.load(Modelset+'/resnet50.pth'))
        net = AKG(args.dataset, backbone, 1024)
        net.to(device)

        CELoss = nn.CrossEntropyLoss()
        GRAPH = GraphLoss(trees, total_nodes, levels, device, args)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

        save_name = args.dataset+'_'+str(args.epoch)+'_'+str(args.batch)+'_'+str(args.img_size)+'_'+str(args.proportion)+'_ResNet-50_'+'_'+args.lr_adjt
        train(nb_epoch, net, trainloader, valloader, testloader, optimizer, scheduler, CELoss, GRAPH, device, args.device, save_name, EXP_number, exp_ROOT, args)
    