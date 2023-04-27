import argparse
import os
# import ruamel_yaml as yaml
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchmetrics import MetricTracker, F1Score, Accuracy, Recall, Precision, Specificity, ConfusionMatrix


from utils import train_utils as utils
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, get_constant_schedule_with_warmup, AutoConfig

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders  

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_title', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(text, book_type) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
         
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=512, return_tensors="pt").to(device) 

        # book_type = torch.tensor(book_type).to(device) 

        outputs = model(input_ids=text_input.input_ids, attention_mask=text_input.attention_mask, labels=book_type, return_dict=True)                  
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_title=loss.item())
        # metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    start_time = time.time() 

    test_f1= F1Score(task="multiclass", num_classes=24, average="macro").to(device)  # F1 score
    test_acc= Accuracy(task="multiclass", num_classes=24, average="micro").to(device)  # Accuracy
    test_rcl= Recall(task="multiclass", num_classes=24, average="macro").to(device)  # Recall
    test_pcl= Precision(task="multiclass", num_classes=24, average="macro").to(device)  # Precision
    # test_sen= mySensitivity(task="multiclass", num_classes=2, average="macro")  # my Sensitivity
    test_spc= Specificity(task="multiclass", num_classes=24, average="macro").to(device) # Specificity
    test_conf_mat= ConfusionMatrix(task="multiclass", num_classes=24).to(device)  # Confusion Matrix

    
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('loss_title', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # # metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Val : '
    print_freq = 50

    
    for i,(text, book_type) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        text_input = tokenizer(text, padding='longest',truncation=True, max_length=512, return_tensors="pt").to(device)  
        book_type = book_type.to(device) 
        

        outputs = model(input_ids=text_input.input_ids, attention_mask=text_input.attention_mask, labels=book_type, return_dict=True)                  
        title_preds = outputs.logits.argmax(-1)
        title_labels = torch.tensor(book_type, dtype=torch.long).to(device)

        test_f1(title_preds, title_labels)
        test_acc(title_preds, title_labels)
        test_spc(title_preds, title_labels)
        test_rcl(title_preds, title_labels)
        test_pcl(title_preds, title_labels)
                 
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    f1_score = test_f1.compute()
    print(f"F1-score: {f1_score}")
    print(f"Accuracy: {test_acc.compute()}")
    print(f"Specificity: {test_spc.compute()}")
    print(f"recall: {test_rcl.compute()}")
    print(f"Precision: {test_pcl.compute()}")
    print(f"Confusion Matric: {test_conf_mat.compute()}")

    del test_f1
    del test_acc
    del test_spc
    del test_rcl
    del test_pcl
    del test_conf_mat

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return f1_score


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    import math
    print("Creating retrieval dataset")
    data = json.load(open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef_itm.json', 'r'))
    n_train = math.floor(len(data) * 0.9)

    train_dataset = [(item['caption'][10:-20], int(item['image'].split('_')[0])-1) for item in data[:n_train]]
    val_dataset = [(item['caption'], int(item['image'].split('_')[0])-1) for item in data[n_train:]]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None]
    
    train_loader, val_loader = create_loader([train_dataset, val_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']],
                                                          num_workers=[0,0],
                                                          is_trains=[True, False], 
                                                          collate_fns=[None,None])   
       
    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    #### Model #### 
    print("Creating model")
    model_config = AutoConfig.from_pretrained(args.text_encoder)
    model_config.num_labels = 24
    model = BertForSequenceClassification.from_pretrained(args.text_encoder, config=model_config)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu')  
        state_dict = checkpoint['model']
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=2e-2, amsgrad=False)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=arg_sche['warmup_epochs'])
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        f1_score = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        if args.evaluate: 
            break
    
        if utils.is_main_process():  
      
            # val_result = title_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            # print(val_result)
            # test_result = title_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)    
            # print(test_result)
            
            # if args.evaluate:                
            #     log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
            #                  **{f'test_{k}': v for k, v in test_result.items()},                  
            #                  'epoch': epoch,
            #                 }
            #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            #         f.write(json.dumps(log_stats) + "\n")     
            # else:
            #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #                  **{f'val_{k}': v for k, v in val_result.items()},
            #                  **{f'test_{k}': v for k, v in test_result.items()},                  
            #                  'epoch': epoch,
            #                 }
            #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            #         f.write(json.dumps(log_stats) + "\n")   
                    
            if f1_score>best and not args.evaluate:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = f1_score    
                best_epoch = epoch
                    
        
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
