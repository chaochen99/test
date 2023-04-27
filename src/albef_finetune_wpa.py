# -*-coding:utf-8-*-

import argparse
import os
import pickle5 as pickle
import math
import contextlib
import random 

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig
from model.tokenization_layoutlmv3_albef import LayoutLMv3Tokenizer_cn
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from model import My_DataLoader_albef_wpa as My_DataLoader
# from model.LayoutLMv3forWPA import LayoutLMv3ForPretraining
# from utils.slack import notification_slack
from utils import train_utils
from torchmetrics import MetricTracker, F1Score, Accuracy, Recall, Precision, Specificity, ConfusionMatrix, MetricCollection
import os
os.environ["USE_TORCH"]= "True"
import time
import datetime
import ruamel.yaml as yaml

import sys
sys.path.append("/nlp_group/wuxing/ALBEF/")
from models.model_retrieval_wpa import ALBEF
from models.vit import interpolate_pos_embed

#再現性
seed = 3407
def fix_seed(seed):
    # random
    random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@contextlib.contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def temp_random_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)

def plot_graph(args, epoch, iter_list, train_losses, val_losses):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.plot(iter_list, train_losses)
    plt.plot(iter_list, val_losses)
    plt.legend(["train_loss", "valid_loss"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/loss.png")
    plt.close()

def plot_graph_2(args, epoch, iter_list, wpa_losses):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.plot(iter_list, wpa_losses)
    plt.legend(["WPA"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/indiv_loss.png")
    plt.close()

def plot_graph_3(args, epoch, iter_list, acces):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.plot(iter_list, acces)
    plt.legend(["acc"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/acces.png")
    plt.close()

def save_hparams(args):
    with open(f"{args.output_model_dir}hparams.txt", mode="w") as f:
        f.writelines(str(args.__dict__))

#save fun 
def save_loss_epcoh(args, model, epoch, iter_list, train_losses, valid_losses, \
    wpa_losses, acces, optimizer, scheduler):
    #os.makedirs pyenv cannot change working directory to .... ↓
    os.makedirs(f"{args.output_model_dir}epoch_{epoch}", exist_ok = True)
    plot_graph(args, epoch, iter_list, train_losses, valid_losses) 
    plot_graph_2(args, epoch, iter_list, wpa_losses)
    plot_graph_3(args, epoch, iter_list, acces)
    save_obj = {
        "epoch": epoch,
        "iter_list": iter_list,
        "train_loss_list": train_losses,
        "valid_loss_list": valid_losses,
        "wpa_losses": wpa_losses,
        "acces": acces,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(
    save_obj,
    f"{args.output_model_dir}epoch_{epoch}/checkpoint.pth",
    )
    del save_obj
    # notification_slack(f"epoch:{epoch}が終了しました。valid_lossは{valid_losses[-1]}です。")
         

def main(args):
    print(args, flush=True)
    if not torch.cuda.is_available():
        raise ValueError("GPU is not available.")
    train_utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + train_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    tokenizer = LayoutLMv3Tokenizer_cn.from_pretrained("hfl/chinese-roberta-wwm-ext", apply_ocr=False)
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    save_hparams(args)

    config = yaml.load(open('/nlp_group/wuxing/ALBEF/configs/retrieval_wpa.yaml', 'r'), Loader=yaml.Loader)
    model = ALBEF(config=config, text_encoder='hfl/chinese-roberta-wwm-ext', tokenizer=tokenizer)
    
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location='cpu')  
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.pretrained)
        print(msg)  
    
    #modelをGPUへ
    model = model.to(device)  
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

    #optimizer 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2, betas=(0.9, 0.98))
    if not args.model_params is None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    #load input_file
    data = []
    # input_names = os.listdir(args.input_file)
    input_names = ['wpa.pkl']
    if args.datasize is not None:
        input_names = input_names[:args.datasize]
    # notification_slack(f"input_file_length: {len(input_names)}")
    for file_name in input_names:
        with open(f"{args.input_file}{file_name}", "rb") as f:
            d = pickle.load(f)
            data += d
    # notification_slack(f"pretraing: datasize is {len(data)}")
    #divide into train and valid
    n_train = math.floor(len(data) * args.ratio_train)
    train_data = data[:n_train]
    valid_data = data[n_train:]
    
    num_tasks = train_utils.get_world_size()
    global_rank = train_utils.get_rank()
    samplers = train_utils.create_sampler([train_data, valid_data], [True, False], num_tasks, global_rank)
    # notification_slack(f"pretraing: train_data is {len(train_data)}, valid_data is {len(valid_data)}.")
    #create dataloader
    my_dataloader = My_DataLoader.My_Dataloader(vocab, random)
    dataloaders = my_dataloader([train_data, valid_data],samplers,batch_size=[args.batch_size, args.batch_size], num_workers=[0, 0], is_trains=[True, False])
    train_dataloader = dataloaders[0]
    valid_dataloader = dataloaders[1]

    #scheduler warm up lineary over fist 0.4% step
    iter_per_epoch = len(train_dataloader)
    num_warmup_steps = round((iter_per_epoch * args.max_epochs) * 0.048)
    if not args.model_params is None:
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("not scheduler", flush = True)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
    
    #define caluculation ml?loss
    def cal_wpa_loss(wpa_logits, batch):
        w_logits = wpa_logits[:,:512]
        
        t  = []
        for i in range(wpa_logits.shape[0]):
            bool_index = torch.ones(512)
            # bool_index[batch["ml_position"][i]] = 0
            bool_index = bool_index * batch["attention_mask"][i]
            t.append(bool_index)
        bool_indexes = torch.stack(t).to(torch.bool)
        predict_label = w_logits[bool_indexes]
        labels = batch["alignment_labels"][bool_indexes].to(torch.long)
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(predict_label + 1e-12, labels)
        acc = (predict_label.argmax(-1) == labels).sum() / labels.shape[0]

        return loss, acc, predict_label , labels
    
    #validation step
    def validation():
        model.eval()
        losses = []
        wpa_losses = []
        acces = []
        metric_collection = MetricCollection([
            F1Score(task="multiclass", num_classes=2, average="macro").to(device),
            Accuracy(task="multiclass", num_classes=2, average="micro").to(device),
            Recall(task="multiclass", num_classes=2, average="macro").to(device),
            Precision(task="multiclass", num_classes=2, average="macro").to(device),
            Specificity(task="multiclass", num_classes=2, average="macro").to(device),
            ConfusionMatrix(task="multiclass", num_classes=2).to(device)  # Confusion Matrix
        ])
        with torch.no_grad():
            val_header = 'Val Epoch: [{}]'.format(epoch)
            val_print_freq = 50   
            val_metric_logger = train_utils.MetricLogger(delimiter="  ")
            val_metric_logger.add_meter('wpa_loss', train_utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            val_metric_logger.add_meter('val_loss', train_utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            val_metric_logger.add_meter('acc', train_utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            for batch in val_metric_logger.log_every(valid_dataloader, val_print_freq, val_header):
                # print(batch['ml_position'])
                inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "pixel_values", "attention_mask", "bool_mi_pos"]}
                text_logits = model.forward(**inputs)
                wpa_loss, acc, wpa_logits, labels = cal_wpa_loss(text_logits, batch)
                val_loss = wpa_loss
                losses.append(val_loss.item())
                wpa_losses.append(wpa_loss.item())
                acces.append(acc.item())
                val_metric_logger.update(wpa_loss=wpa_loss.item())
                val_metric_logger.update(val_loss=val_loss.item())
                val_metric_logger.update(acc=acc.item())
                # wpa_logits = wpa_logits.view(-1, 2)
                # labels= labels.view(-1)
                metric_collection(wpa_logits, labels)
            ave_losses = sum(losses) / len(losses)
            ave_wpa = sum(wpa_losses) / len(wpa_losses)
            ave_acc = sum(acces) / len(acces)
            model.train()
            metrics = metric_collection.compute()
            print(metrics)
            del metric_collection
            return ave_losses, ave_wpa, ave_acc, metrics['MulticlassF1Score'].item()
    
    train_losses = []
    valid_losses = []
    wpa_losses = []
    acces = []
    iter_list = []
    ##epcoh
    if not args.model_params is None:
        epochs = range(checkpoint["epoch"] +1, args.max_epochs)
        train_losses = checkpoint["train_loss_list"]
        valid_losses = checkpoint["valid_loss_list"]
        iter_list = checkpoint["iter_list"]
        wpa_losses = checkpoint["wpa_losses"]
        acces = checkpoint["acces"]
        # iter_list = [0, 1314, 2628, 3942, 5265, 6
        #570, 7884, 9198, 10512, 11826, 13140,13141, 14455, 15769, 17083, 18397, 19711, 21025, 22339, 23653, 24967, 26281]
        print(epochs, flush=True)
        print(train_losses, flush=True)
        print(len(iter_list),iter_list, flush=True)
    else:
        epochs = range(args.max_epochs)
    
    # notification_slack("start training!")
    iter_per_epoch = len(train_dataloader)
    print("iter: ", epochs[0] * iter_per_epoch, flush=True)
    model.train()
    scaler = GradScaler() 
    print("Start training")
    start_time = time.time() 

    best = 0
    best_epoch = 0  
    
    for epoch in epochs:
        train_dataloader.sampler.set_epoch(epoch)
        
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50   
        metric_logger = train_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', train_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('wpa_loss', train_utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('train_loss', train_utils.SmoothedValue(window_size=1, fmt='{value:.4f}')) 
        
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50
        for i, batch in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
            # print(batch['ml_position'])
            iter = epoch * iter_per_epoch + i
            inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "pixel_values", "attention_mask", "bool_mi_pos"]}
            with autocast():
                wpa_logits = model(**inputs)
                wpa_loss, _, _, _ = cal_wpa_loss(wpa_logits, batch)
                loss = wpa_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()
            optimizer.zero_grad()
            metric_logger.update(wpa_loss=wpa_loss.item() if isinstance(wpa_loss, torch.Tensor) else wpa_loss)
            metric_logger.update(train_loss=loss.item() if isinstance(loss, torch.Tensor) else loss)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iter_list.append(iter)
        train_losses.append(loss.item())
        val_loss, val_wpa_loss, val_acc, val_f1 = validation()
        valid_losses.append(val_loss)
        wpa_losses.append(val_wpa_loss)  
        acces.append(val_acc)     
        print(
            f"epoch:{epoch}, iter:{iter}, {i},  train_loss: {loss.item()}, valid_loss: {val_loss}, acc:{str(val_acc)}"
            )
            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())     

        if train_utils.is_main_process() and val_f1>best:  
            save_loss_epcoh(
                args = args,
                model = model,
                epoch = epoch,
                iter_list = iter_list,
                train_losses = train_losses, 
                valid_losses = valid_losses, 
                wpa_losses = wpa_losses,
                acces = acces,
                optimizer = optimizer, 
                scheduler = scheduler,
                )
            print("epoch", epoch, loss.item(), flush=True)
            best=val_f1
            best_epoch = epoch
        dist.barrier()     
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
        
    # save_hparams(args)
    # notification_slack("学習が無事に終わりました。")
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_params", type=str)
    parser.add_argument("--output_model_dir", type=str, required=True)
    # parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--ratio_train", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--datasize", type=int)
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    main(args)


