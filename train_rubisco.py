# import numpy as np
# import torch
# from scipy.stats import pearsonr
#
# from dataset.load_dataset import load_dataset, Smiles_data, load_rubisco
# from dataset.load_dataset import load_rubisco
# from models.rubisco_model import rubisco
# import torch.nn.functional as F
#
# def train_model(loader, model, optimizer, device, kcat_scaler):
#     model.train()
#     optimizer.zero_grad()
#     pred_kcats = []
#
#     true_kcats = []
#
#     losses = 0
#     batch_count = 0
#     for i, batch in enumerate(loader):
#         true_kcat = batch[0]
#         # true_km = batch[1]
#         substrate = batch[2].to(device)
#         product = batch[3].to(device)
#         substrate_embedding = batch[4].to(device)
#         product_embedding = batch[5].to(device)
#         condition = batch[6].to(device)
#         # protein = batch[4].to(device)
#         pred_kcat = model(substrate, product, substrate_embedding, product_embedding, condition)
#         loss = F.mse_loss(true_kcat.cpu(), pred_kcat.cpu())
#         loss.backward()
#         optimizer.step()
#         pred_kcats.extend(pred_kcat.detach().cpu().numpy())
#         # pred_kms.extend(pred_km.detach().cpu().numpy())
#         true_kcats.extend(true_kcat.detach().cpu().numpy())
#         # true_kms.extend(true_km.detach().cpu().numpy())
#         losses += loss.item()
#         batch_count += 1
#     kcat = kcat_scaler.inverse_transform(np.array(true_kcats).reshape(-1, 1)).squeeze(1)
#     _kcat = kcat_scaler.inverse_transform(np.array(pred_kcats).reshape(-1, 1)).squeeze(1)
#     kcat_pcc, kcat_p_value = pearsonr(kcat, _kcat)
#     # print(f"train_kcat_pcc: {kcat_pcc}")
#     # print(f"loss: {losses / batch_count}")
#
# @torch.no_grad()
# def eval_model(loader, model, device, kcat_scaler):
#     pred_kcats = []
#     true_kcats = []
#     losses = 0
#     batch_count = 0
#     for i, batch in enumerate(loader):
#         true_kcat = batch[0]
#         # true_km = batch[1]
#         substrate = batch[2].to(device)
#         product = batch[3].to(device)
#         substrate_embedding = batch[4].to(device)
#         product_embedding = batch[5].to(device)
#         condition = batch[6].to(device)
#         # protein = batch[4].to(device)
#         pred_kcat = model(substrate, product, substrate_embedding, product_embedding, condition)
#         loss = F.mse_loss(true_kcat.cpu(), pred_kcat.cpu())
#         pred_kcats.extend(pred_kcat.detach().cpu().numpy())
#         # pred_kms.extend(pred_km.detach().cpu().numpy())
#         true_kcats.extend(true_kcat.detach().cpu().numpy())
#         # true_kms.extend(true_km.detach().cpu().numpy())
#         losses += loss.item()
#         batch_count += 1
#     kcat = kcat_scaler.inverse_transform(np.array(true_kcats).reshape(-1, 1)).squeeze(1)
#     _kcat = kcat_scaler.inverse_transform(np.array(pred_kcats).reshape(-1, 1)).squeeze(1)
#     kcat_pcc, kcat_p_value = pearsonr(kcat, _kcat)
#     print(f"test_kcat_pcc: {kcat_pcc}")
#     # print(f"eval loss: {losses / batch_count}")
#
# def main():
#     train_dataloader, test_dataloader, kcat_scaler, km_scaler = load_rubisco("smiles")   ### 修改
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = rubisco(input_features=512, model_name="smiles").to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
#     for i in range(500):
#         train_model(train_dataloader, model,optimizer, device, kcat_scaler)
#         eval_model(test_dataloader, model, device, kcat_scaler)
#
#
#
#
# if __name__ == '__main__':
#     # # 多gpu运行命令： CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_test_model.py --multi_gpu True
#     # parser.add_argument("--gpu", type=bool, default=True)
#     # parser.add_argument("--multi_gpu", type=bool, default=False,
#     #                     help="use multi gpu. If you want to use multi gpu for training, "
#     #                          "you should use this command:"
#     #                          "CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_test_model.py --multi_gpu True")
#     # parser.add_argument('--gpu_id', type=str, default='0')  # 单gpu，则gpu_id=0
#     # parser.add_argument("--batch_size", type=str, default="256")
#     # parser.add_argument("--model_name", type=str, default='fusion')  # 3dgnn带有边特征
#     # parser.add_argument("--result_dir", type=str, default='best')
#     # # predict/km_predict
#     # # predict/Synechocystis sp
#     # # predict/rubisco
#     # parser.add_argument("--path", type=str, default="predict/rubisco")  # predict/km_predict
#     # args = parser.parse_args()
#     main()


import warnings

warnings.filterwarnings('ignore')  # 忽略所有警告

from torch import nn
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import os
import pandas as pd
import torch
from dataset.load_dataset import load_dataset, Smiles_data, load_rubisco
import argparse
from models.our_model import Model
from utils import train_model, Seed_everything, eval_model, metrics, load_uni_modal_pretrain
from utils import EarlyStopping
from datetime import datetime
from accelerate.utils import set_seed

parser = argparse.ArgumentParser()


def replace_syncbn_with_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.SyncBatchNorm):
            num_features = module.num_features
            # 替换为普通的 BatchNorm
            setattr(model, name, nn.BatchNorm2d(num_features))
        else:
            # 递归处理子模块
            replace_syncbn_with_bn(module)


def main(args):
    train_losses = []
    val_losses = []
    train_kcat_pccs = []
    val_kcat_pccs = []
    train_km_pccs = []
    val_km_pccs = []
    last_epoch = 1
    epoch = 0

    if args.multi_gpu:
        set_seed(int(args.seed))
    else:
        Seed_everything(seed=int(args.seed))
    # 获取当前时间
    now = datetime.now()
    # 按照指定格式输出时间
    formatted_time = now.strftime('%Y_%m_%d_%H_%M')
    result_path = os.path.join("result", args.model_name)
    accelerator = None
    device = 'cpu'
    lr = float(args.lr)
    if args.model_name == "smiles":
        modal_name = 'smiles_extractor.'
    elif args.model_name == "protein":
        modal_name = 'protein_extractor.'
    elif args.model_name == "conditions":
        modal_name = 'llm_extractor.'
    elif args.model_name == "fusion":
        modal_name = '*'
    else:
        raise ValueError("Error modal name !!!")
    # start a new wandb run to track this script
    print(args.model_name)
    if args.gpu:
        if not args.multi_gpu:  # 单卡训练
            device = torch.device('cuda:' +  # 设置设备，如果可用并且指定了 GPU，则使用 GPU，否则使用 CPU。
                                  args.gpu_id if torch.cuda.is_available() and args.gpu else 'cpu')
            print(f"using single gpu: {device} !!!")
            if "rubisco_km" in args.path:
                model = Model(input_features=512, device=device, model_name=args.model_name, rubisco_km=True,
                              rubisco_kcat=False).to(device)
            else:
                model = Model(input_features=512, device=device, model_name=args.model_name, rubisco_km=False,
                              rubisco_kcat=True).to(device)
            wandb.init(
                # set the wandb project where this run will be logged
                project="my-awesome-project",

                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model_name,
                    "epochs": args.epochs,
                }
            )
        else:  # 多卡训练
            kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(kwargs_handlers=[kwargs])
            device = accelerator.device
            if "rubisco_km" in args.path:
                model = Model(input_features=512, device=device, model_name=args.model_name, rubisco_km=True,
                              rubisco_kcat=False)
            else:
                model = Model(input_features=512, device=device, model_name=args.model_name, rubisco_km=False,
                              rubisco_kcat=True)
            # 确保 model 中没有 SyncBN
            # replace_syncbn_with_bn(model)
            wandb.init(
                # set the wandb project where this run will be logged
                project="my-awesome-project",
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model_name,
                    "epochs": args.epochs,
                }
            )
    else:
        if "rubisco_km" in args.path:
            model = Model(input_features=512, device=device, model_name=args.model_name, rubisco_km=True,
                          rubisco_kcat=False)
        else:
            model = Model(input_features=512, device=device, model_name=args.model_name, rubisco_km=False,
                          rubisco_kcat=True)
        print(f"cpu train: {device} !!!")
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",

            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.model_name,
                "epochs": args.epochs,
            }
        )
    my_print = accelerator.print if args.multi_gpu else print  # 设置print函数

    if (not args.multi_gpu) or (args.multi_gpu and accelerator.is_main_process):  # 只有单线程程序、多gpu程序中的主线程才能进入
        if args.task == "train":  # train建立结果目录
            # 注意：只有主线程的result_path是os.path.join(result_path, formatted_time)，其他分线程不是，涉及到文件操作一定要只有主线程进入，否则报错
            result_path = os.path.join(result_path, formatted_time)
            if not os.path.exists(result_path):
                my_print("path not exists, make a new train dir!!!")
                os.makedirs(result_path)
            # 保存参数
            args_dict = vars(args)
            # 将参数字典写入到文本文件
            with open(os.path.join(result_path, f'{formatted_time}_hyperparameter.txt'), 'w') as file:
                for key, value in args_dict.items():
                    if value is not None:  # 只写入有值的参数
                        file.write(f"{key}={value}\n")
        else:
            if os.path.exists(os.path.join(result_path, args.result_dir)):
                result_path = os.path.join(result_path, args.result_dir)
                if args.task == "continue_train":
                    # 将参数字典写入到文本文件
                    args_dict = vars(args)
                    with open(os.path.join(result_path, f'{formatted_time}_continue_train_hyperparameter.txt'),
                              'w') as file:
                        for key, value in args_dict.items():
                            if value is not None:  # 只写入有值的参数
                                file.write(f"{key}={value}\n")
            else:
                raise FileNotFoundError("找不到文件")

    train_dataloader, val_dataloader, test_dataloader, kcat_scaler, km_scaler = load_rubisco(args.model_name,
                                                                                             path=args.path, batch_size=int(args.batch_size))
    early_stopping = EarlyStopping(patience=500, verbose=True, trace_func=my_print,
                                   path=result_path, multi_gpu=args.multi_gpu, accelerator=accelerator)  # 使用早停策略
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=1e-6)

    if args.model_name == "fusion" and args.task == "train":  # 加载预训练权重，预训练权重由于是分开训练的，需要单独加载，加载后则跳过加载全部权重
        load_uni_modal_pretrain(model, pth_file='models/pt_file/protein_pretrain.pt',
                                use_multi_gpu=True, modal_name='protein_extractor.', device=device,
                                my_print=my_print, frozen=True)
        load_uni_modal_pretrain(model, pth_file='models/pt_file/llm_pretrain.pt',
                                use_multi_gpu=False, modal_name='llm_extractor.', device=device,
                                my_print=my_print, frozen=True)
        load_uni_modal_pretrain(model, pth_file='models/pt_file/smiles_pretrain.pt',
                                use_multi_gpu=False, modal_name='smiles_extractor.', device=device,
                                my_print=my_print, frozen=True)
    if args.task == "test" or args.task == "continue_train":  # 如果已有权重且开启测试或继续训练模式，则读取权重
        my_print("loading pretrain weight!!!!")
        if (not args.multi_gpu) or (args.multi_gpu and accelerator.is_main_process):  # protein_pretrain.pt  last_weight.pt
            checkpoint = torch.load(os.path.join(result_path, "protein_pretrain.pt"), map_location=device)
            load_uni_modal_pretrain(model, pth_file=os.path.join(result_path, "protein_pretrain.pt"),
                                    use_multi_gpu=checkpoint['use_multi_gpu'], modal_name=modal_name, device=device,
                                    my_print=my_print, frozen=False)

            if args.task == "continue_train":
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    raise RuntimeError("模型参数与optimizer不严格一致，无法加载optimizer参数继续训练")
            epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            early_stopping.val_loss_min = 1e6
            early_stopping.best_val_loss = 1e6
            train_kcat_pccs = checkpoint['train_kcat_pccs']
            val_kcat_pccs = checkpoint['val_kcat_pccs']
            train_km_pccs = checkpoint['train_km_pccs']
            val_km_pccs = checkpoint['val_km_pccs']

    if args.multi_gpu:
        os.environ['NCCL_P2P_DISABLE'] = '1'
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,  ####修改
            test_dataloader
        )

    if args.task != "test":  # 训练模型
        my_print("start train!!!")

        for epoch in range(last_epoch, int(args.epochs) + last_epoch):
            my_print(f"Epoch: {epoch}")
            train_loss, train_kcat_pcc, train_kcat_p_value, train_km_pcc, train_km_p_value = train_model(
                train_dataloader,
                model,
                optimizer, scheduler,
                kcat_scaler,
                km_scaler,
                epoch,
                args, accelerator=accelerator, device=device)
            my_print(f"train_loss: {train_loss}, train_kcat_pcc: {train_kcat_pcc}, train_km_pcc: {train_km_pcc}, "
                     f"train_kcat_p_value: {train_kcat_p_value}, train_km_p_value: {train_km_p_value}")
            train_losses.append(train_loss)
            train_kcat_pccs.append(train_kcat_pcc)
            train_km_pccs.append(train_km_pcc)

            val_loss, val_kcat_pcc, val_kcat_p_value, val_km_pcc, val_km_p_value = eval_model(val_dataloader,
                                                                                              model,
                                                                                              kcat_scaler, km_scaler,
                                                                                              args, is_test=False,
                                                                                              accelerator=accelerator,
                                                                                              device=device)
            my_print(f"val_loss: {val_loss}, val_kcat_pcc: {val_kcat_pcc}, val_km_pcc: {val_km_pcc}, "
                     f"val_kcat_p_value: {val_kcat_p_value}, val_km_p_value: {val_km_p_value}")
            val_losses.append(val_loss)
            val_kcat_pccs.append(val_kcat_pcc)
            val_km_pccs.append(val_km_pcc)
            early_stopping(val_loss, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), epoch,
                           train_losses,
                           val_losses, train_kcat_pccs, val_kcat_pccs, train_km_pccs, val_km_pccs, accelerator)
            # scheduler.step(val_loss)
            if (not args.multi_gpu) or (args.multi_gpu and accelerator.is_main_process):
                wandb.log({"train_kcat_pcc": train_kcat_pcc,
                           "train_km_pcc": train_km_pcc,
                           "train_loss": train_loss,
                           "val_kcat_pcc": val_kcat_pcc,
                           "val_km_pcc": val_km_pcc,
                           "val_loss": val_loss
                           })
    if (not args.multi_gpu) or (args.multi_gpu and accelerator.is_main_process):
        train_curve = pd.DataFrame({'epoch': epoch,
                                    'train_loss': train_losses,
                                    'val_loss': val_losses,
                                    'train_pcc_kcat': train_kcat_pccs,
                                    'train_pcc_km': train_km_pccs,
                                    'val_pcc_kcat': val_kcat_pccs,
                                    'val_pcc_km': val_km_pccs,
                                    })
        train_curve.to_csv(os.path.join(result_path, '_train_curve_' + str(epoch) + ".csv"),
                           sep=",", index=False)

    my_print("start last weight test")
    test_loss, test_kcat_pcc, test_kcat_p_value, test_km_pcc, test_km_p_value, \
        true_kcats, true_kms, pred_kcats, pred_kms, indexes = eval_model(test_dataloader,
                                                                model,
                                                                kcat_scaler, km_scaler,
                                                                args, is_test=True, accelerator=accelerator,
                                                                device=device)
    my_print(f"test_loss: {test_loss}, test_kcat_pcc: {test_kcat_pcc}, test_kcat_p_value: {test_kcat_p_value}, "
             f"test_km_pcc: {test_km_pcc}, test_km_p_value: {test_km_p_value}")
    if (not args.multi_gpu) or (args.multi_gpu and accelerator.is_main_process):
        metrics(true_kcats, pred_kcats, true_kms, pred_kms,
                result_path=os.path.join(result_path, '_' + str(epoch) + 'test_result' + ".csv"),
                curve_path=os.path.join(result_path, '_' + str(epoch) + 'test_curve_result' + ".csv"),
                test_kcat_pcc=test_kcat_pcc, test_km_pcc=test_km_pcc, indexes=indexes)
        checkpoint = torch.load(os.path.join(result_path, "best.pt"), map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        load_uni_modal_pretrain(model, pth_file=os.path.join(result_path, "best.pt"),
                                use_multi_gpu=checkpoint['use_multi_gpu'], modal_name=modal_name, device=device,
                                my_print=my_print, frozen=False)
        epoch = checkpoint['epoch']

    my_print("start best weight test")

    test_loss, test_kcat_pcc, test_kcat_p_value, test_km_pcc, test_km_p_value, \
        true_kcats, true_kms, pred_kcats, pred_kms, indexes = eval_model(test_dataloader,
                                                                model,
                                                                kcat_scaler, km_scaler,
                                                                args, is_test=True, accelerator=accelerator,
                                                                device=device)

    my_print(f"test_loss: {test_loss}, test_kcat_pcc: {test_kcat_pcc}, test_kcat_p_value: {test_kcat_p_value}, "
             f"test_km_pcc: {test_km_pcc}, test_km_p_value: {test_km_p_value}")
    if (not args.multi_gpu) or (args.multi_gpu and accelerator.is_main_process):
        metrics(true_kcats, pred_kcats, true_kms, pred_kms,
                result_path=os.path.join(result_path, '_best_' + str(epoch) + '_test_result' + ".csv"),
                curve_path=os.path.join(result_path,
                                        '_best_' + str(epoch) + '_test_curve_result' + ".csv"),
                test_kcat_pcc=test_kcat_pcc, test_km_pcc=test_km_pcc, indexes=indexes)
        wandb.finish()


if __name__ == '__main__':
    # 多gpu运行命令： CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 accelerate launch train_rubisco.py --multi_gpu True
    # predict/rubisco_km/dataset
    # predict/rubisco/dataset
    parser.add_argument("--path", type=str, default="predict/rubisco/dataset")
    parser.add_argument("--gpu", type=bool, default=True, help="use gpu or cpu")
    parser.add_argument("--multi_gpu", type=bool, default=False, help="use multi gpu. If you want to use multi gpu for training, "
                                                                      "you should use this command:"
                                                                      "CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_test_model.py --multi_gpu True")
    parser.add_argument('--gpu_id', type=str, default='7', help="single gpu id")
    parser.add_argument("--batch_size", type=str, default="8") # 8 64
    parser.add_argument("--epochs", type=str, default="100")
    parser.add_argument("--lr", type=str, default="0.001")  # 微调使用更小的学习率
    parser.add_argument("--seed", type=str, default="2024")
    parser.add_argument("--model_name", type=str, default='protein',
                        choices=['smiles', 'protein', 'fusion', 'conditions'])
    parser.add_argument("--task", type=str, default='continue_train', choices=['train', 'test', 'continue_train'])
    parser.add_argument("--result_dir", type=str, default='/home/zhangyangyu/kcat_km_predict/result/protein/best')  # best
    # parser.add_argument("--result_dir", type=str, default='/home/zhangyangyu/kcat_km_predict/predict/rubisco_km')  # best
    args = parser.parse_args()
    main(args=args)
