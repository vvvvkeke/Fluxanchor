import math
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, multi_gpu=False,
                 accelerator=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.multi_gpu = multi_gpu
        if self.multi_gpu:
            ## Save the object passed to disk once per machine. Use in place of torch.save.针对的是多机器情形
            self.save = accelerator.save
        else:
            self.save = torch.save

    def __call__(self, val_loss, model, optimizer, scheduler, epoch, train_losses,
                 val_losses, train_kcat_pccs, val_kcat_pccs, train_km_pccs, val_km_pccs, accelerator):
        if self.multi_gpu:  # self.multi_gpu and accelerator.is_main_process
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)  ##将prepare的模型保存回正常的model
        else:
            unwrapped_model = model
        self.save({
            'model_state_dict': unwrapped_model,
            'optimizer_state_dict': optimizer,
            'scheduler_state_dict': scheduler,
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_kcat_pccs': train_kcat_pccs,
            'val_kcat_pccs': val_kcat_pccs,
            'train_km_pccs': train_km_pccs,
            'val_km_pccs': val_km_pccs,
            'use_multi_gpu': self.multi_gpu,
            # 'kcat_r2': kcat_r2,
            # 'km_r2': km_r2
        },
            os.path.join(self.path, "last_weight.pt"))

        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, unwrapped_model, optimizer, scheduler, epoch, train_losses,
                                 val_losses, train_kcat_pccs, val_kcat_pccs, train_km_pccs, val_km_pccs, accelerator)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, unwrapped_model, optimizer, scheduler, epoch, train_losses,
                                 val_losses, train_kcat_pccs, val_kcat_pccs, train_km_pccs, val_km_pccs, accelerator)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, unwrapped_model, optimizer, scheduler, epoch, train_losses,
                        val_losses, train_kcat_pccs, val_kcat_pccs, train_km_pccs, val_km_pccs, accelerator):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.save({
                'model_state_dict': unwrapped_model,
                'optimizer_state_dict': optimizer,
                'scheduler_state_dict': scheduler,
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_kcat_pccs': train_kcat_pccs,
                'val_kcat_pccs': val_kcat_pccs,
                'train_km_pccs': train_km_pccs,
                'val_km_pccs': val_km_pccs,
                'use_multi_gpu': self.multi_gpu
            },
                os.path.join(self.path, "best.pt"))

            self.val_loss_min = val_loss


def load_uni_modal_pretrain(model, pth_file: str, use_multi_gpu: bool, modal_name: str, device, my_print=print,
                            frozen: bool = True):
    if frozen:
        my_print(f"loading {modal_name} weight and frozen it!!!!")
    else:
        my_print(f"loading {modal_name} weight and finetune it!!!!")
    modal = ["protein_extractor.", "llm_extractor.", "smiles_extractor."]
    if modal_name == modal[0]:
        extract = model.protein_extractor
    elif modal_name == modal[1]:
        extract = model.llm_extractor
    elif modal_name == modal[2]:
        extract = model.smiles_extractor
    elif modal_name == "*":
        extract = model
    else:
        raise ValueError("no this pretrain modal!!!")
    # 预训练的权重
    weights = torch.load(pth_file, map_location=device)
    state_dict = {}
    for k, v in weights['model_state_dict'].items():
        name = k[7:] if use_multi_gpu else k  # 多卡训练模型，权重文件会多了module，需要去掉
        if name.startswith(modal_name):
            state_dict[name[len(modal_name):]] = v
        elif modal_name == "*":
            state_dict[name] = v
    extract.load_state_dict(state_dict)
    # 冻结encoder的参数
    if frozen:
        for param in extract.parameters():
            param.requires_grad = False
    # if modal_name == "protein_extractor.":  # 解冻protein模态最后两层
    #     for module in [model.protein_extractor.FC_1, model.protein_extractor.FC_2]:
    #         for p in module.parameters():
    #             p.requires_grad = True
    # if modal_name == "smiles_extractor.":  # 解冻smiles模态最后两层
    #     for module in [model.smiles_extractor.cross_attention.kcat_o, model.smiles_extractor.cross_attention.km_o]:
    #         for p in module.parameters():
    #             p.requires_grad = True
    # if modal_name == "llm_extractor.":  # 解冻llm模态最后两层
    #     for module in model.llm_extractor.task_heads:
    #         for p in module.parameters():
    #             p.requires_grad = True


def inverse_transformation(kcat_scaler, km_scaler, true_kcat: list, true_km: list, pred_kcat: list, pred_km: list):
    kcat = kcat_scaler.inverse_transform(np.array(true_kcat).reshape(-1, 1)).squeeze(1)
    _kcat = kcat_scaler.inverse_transform(np.array(pred_kcat).reshape(-1, 1)).squeeze(1)
    km = km_scaler.inverse_transform(np.array(true_km).reshape(-1, 1)).squeeze(1)
    _km = km_scaler.inverse_transform(np.array(pred_km).reshape(-1, 1)).squeeze(1)

    return kcat, km, _kcat, _km


def get_pcc(true_kcat: list, true_km: list,
            pred_kcat: list, pred_km: list):
    # if task == "kcat_km":  ## [0, 1] -> []
        # 计算皮尔逊相关系数
    kcat_pcc, kcat_p_value = pearsonr(true_kcat, pred_kcat)
    km_pcc, km_p_value = pearsonr(true_km, pred_km)
    return kcat_pcc, kcat_p_value, km_pcc, km_p_value


def train_model(loader, model, optimizer, scheduler,
                kcat_scaler,
                km_scaler,
                epoch,
                args, accelerator, device='cpu'):
    model.train()
    optimizer.zero_grad()
    pred_kcats = []
    pred_kms = []
    true_kcats = []
    true_kms = []
    losses = 0
    batch_count = 0
    disable = False

    if args.multi_gpu:
        disable = not accelerator.is_main_process
    for i, batch in tqdm(enumerate(loader), disable=disable):
        # 梯度累积是一种技术，您可以使用比机器通常能够容纳在内存中的更大的批次大小进行训练。这是通过在多个批次上累积梯度来完成的，并且仅在执行一定数量的批次后才更新优化器。
        if args.multi_gpu:
            with accelerator.accumulate(model):  ###梯度累积

                true_kcat = batch[0]
                true_km = batch[1]
                if args.model_name == "fusion":
                    p_output, s_output, l_output, pred_kcat, pred_km, loss, index = model(batch, args.multi_gpu)
                else:
                    pred_kcat, pred_km, loss, index = model(batch, args.multi_gpu)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                pred_kcat = accelerator.gather_for_metrics(pred_kcat)
                pred_km = accelerator.gather_for_metrics(pred_km)
                true_kcat = accelerator.gather_for_metrics(true_kcat)
                true_km = accelerator.gather_for_metrics(true_km)
            scheduler.step()
        else:
            true_kcat = batch[0].to(device)
            true_km = batch[1].to(device)

            if args.model_name == "fusion":
                p_output, s_output, l_output, pred_kcat, pred_km, loss, index = model(batch, args.multi_gpu)
            else:
                pred_kcat, pred_km, loss, index = model(batch, args.multi_gpu)  # 修改
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        pred_kcats.extend(pred_kcat.detach().cpu().numpy())
        pred_kms.extend(pred_km.detach().cpu().numpy())
        true_kcats.extend(true_kcat.detach().cpu().numpy())
        true_kms.extend(true_km.detach().cpu().numpy())
        losses += loss.item()
        batch_count += 1
    true_kcats, true_kms, pred_kcats, pred_kms = inverse_transformation(kcat_scaler, km_scaler, true_kcats, true_kms,
                                                                        pred_kcats, pred_kms)
    kcat_pcc, kcat_p_value, km_pcc, km_p_value = get_pcc(true_kcats, true_kms, pred_kcats, pred_kms)


    return losses / batch_count, kcat_pcc, kcat_p_value, km_pcc, km_p_value


@torch.no_grad()
def eval_model(loader, model,
               kcat_scaler,
               km_scaler,
               args, is_test=True, accelerator=None, device='cpu'):
    model.eval()
    pred_kcats = []
    pred_kms = []
    true_kcats = []
    true_kms = []
    indexes = []
    losses = 0
    batch_count = 0
    disable = False
    if args.multi_gpu:
        disable = not accelerator.is_main_process
    for i, batch in tqdm(enumerate(loader), disable=disable):
        true_kcat = batch[0]
        true_km = batch[1]
        if args.model_name == "fusion":
            p_output, s_output, l_output, pred_kcat, pred_km, loss, index = model(batch, args.multi_gpu)

        else:
            pred_kcat, pred_km, loss, index = model(batch, args.multi_gpu)  # 修改
        indexes.extend(index.cpu().numpy())
        if args.multi_gpu:
            pred_kcat = accelerator.gather_for_metrics(pred_kcat)
            pred_km = accelerator.gather_for_metrics(pred_km)
            true_kcat = accelerator.gather_for_metrics(true_kcat)
            true_km = accelerator.gather_for_metrics(true_km)
        else:
            true_kcat = true_kcat.to(device)
            true_km = true_km.to(device)

        pred_kcats.extend(pred_kcat.detach().cpu().numpy())
        pred_kms.extend(pred_km.detach().cpu().numpy())
        true_kcats.extend(true_kcat.detach().cpu().numpy())
        true_kms.extend(true_km.detach().cpu().numpy())
        losses += loss.item()
        batch_count += 1

    true_kcats, true_kms, pred_kcats, pred_kms = inverse_transformation(kcat_scaler, km_scaler, true_kcats, true_kms,
                                                                        pred_kcats, pred_kms)
    kcat_pcc, kcat_p_value, km_pcc, km_p_value = get_pcc(true_kcats, true_kms, pred_kcats, pred_kms)

    if is_test:
        return losses / batch_count, kcat_pcc, kcat_p_value, km_pcc, km_p_value, true_kcats, true_kms, pred_kcats, pred_kms, indexes
    else:
        return losses / batch_count, kcat_pcc, kcat_p_value, km_pcc, km_p_value


def Seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_directory_if_not_exists(file_path):
    """
    Summary:该函数用于检查输入的目录路径是否存在, 若存在则不做任何操作, 若不存在则创建
    """

    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"Directory {file_path} has been created.")
    else:
        print(f"Directory {file_path} already exists.")

def metrics(true_kcats, pred_kcats, true_kms, pred_kms, result_path, curve_path, test_kcat_pcc, test_km_pcc, indexes):
    kcat_r2 = r2_score(true_kcats, pred_kcats)
    km_r2 = r2_score(true_kms, pred_kms)
    kcat_mse = np.mean((true_kcats - pred_kcats) ** 2)
    km_mse = np.mean((true_kms - pred_kms) ** 2)
    kcat_mae = np.mean(np.abs(true_kcats - pred_kcats))
    km_mae = np.mean(np.abs(true_kms - pred_kms))
    kcat_rmse = np.sqrt(kcat_mse)
    km_rmse = np.sqrt(km_mse)
    eval_result = pd.DataFrame({'kcat_r2': [kcat_r2],
                                'km_r2': [km_r2],
                                'kcat_mse': [kcat_mse],
                                'km_mse': [km_mse],
                                'kcat_mae': [kcat_mae],
                                'km_mae': [km_mae],
                                'kcat_rmse': [kcat_rmse],
                                'km_rmse': [km_rmse],
                                'test_pcc_kcat': test_kcat_pcc,
                                'test_pcc_km': test_km_pcc
                                })
    eval_result.to_csv(
        result_path,
        sep=",", index=False)

    test_result = pd.DataFrame({'index': indexes,
                                'true_kcats': true_kcats,
                                'true_kms': true_kms,
                                'pred_kcats': pred_kcats,
                                'pred_kms': pred_kms
                                })
    test_result.to_csv(
        curve_path,
        sep=",", index=False)
