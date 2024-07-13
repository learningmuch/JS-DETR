# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch,evaluate_test
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    #  这里是每次换了需要改的几个地方
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume', default='./weights/detr_resnet50.pth', help='resume from checkpoint')
    # 权重输出位置。
    parser.add_argument('--output_dir', default='./weight_all/result',
                        help='path where to save, empty for no saving')

    # parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    # parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
    #                     help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    # parser.add_argument('--coco_path', type=str,default='./data/coco')
    parser.add_argument('--coco_path', type=str,default=r'E:\DETR\detr.6.21\data\coco')
    # parser.add_argument('--coco_path', type=str,default='H:\datasets\augment\DETR\coco')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # 获取当前目录
    current_directory = os.getcwd()

    # 创建结果文件夹，如果不存在的话
    if not os.path.exists('results'):
        os.makedirs('results')

        # 构造文件名
    file_name = "test{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 构造完整文件路径
    results_test_file = os.path.join(current_directory, 'results', file_name)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ipdb.set_trace()
    # 损失函数在这里修改
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # 它结合了Adam优化器和权重衰减（weight decay）的方法，能够有效地调整模型的参数，并减少过拟合的风险。
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # 动态调整学习率的大小
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # 在这里面有个数据增强的操作。在coco.py里面make_coco_transforms 通过随机翻转、随机调整尺度、随机裁剪和归一化等操作
    dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)
    # 分布式训练 多个gpu使用，这里我们用不到
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        # sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # 通过 batch_sampler_train 对训练数据集进行批次采样后，每次迭代时可以得到一个批次的数据，用于模型训练
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # 创建一个训练数据集的数据加载器，可以从数据集中按照设定的 batch 大小生成 batch，并对 batch 中的多个样本进行拼接和填充。同时，可以设置多个子进程来加速数据加载。
    # 1.把不符合的数据全部丢弃。不要遮挡的那些物体，2.对宽高进行处理、ground truth 宽和高不要超过图片的宽高，3.把batchsize里面不同图片大小统一成一个大小、针对填充区域，用true来表达，
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        # base_ds = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)


    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 在原来的基础上加了strict=False
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # model_without_ddp.load_state_dict(checkpoint['model'])
        # 在resume模式里面，如果不是进入了验证模式，即在训练模式下，则 start_epoch 被设置为上一次训练的 epoch 加 1，以继续训练。
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # 添加测试集功能。
    if args.test:
        test_stats_test, coco_evaluator_test = evaluate_test(model, criterion, postprocessors,
                                              data_loader_test, base_ds_test, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator_test.coco_eval["bbox"].eval, output_dir / "test.pth")
        if coco_evaluator_test is not None:
            (output_dir / 'test').mkdir(exist_ok=True)
            if "bbox" in coco_evaluator_test.coco_eval:
                with open(results_test_file, "a") as f:
                    # 写入的数据包括coco指标还有loss和learning rate
                    result_test_info = [f"{i:.4f}" for i in coco_evaluator_test.coco_eval["bbox"].stats]
                    ## 打印coco_evaluator_test.coco_eval["bbox"].stats 的内容看看
                performance = float(result_test_info[1])
                print("ap50的值是", performance)  # 输出性能指标
        return

    print("Start training")
    start_time = time.time()
    best_performance = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # 多GPU使用
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            #  checkpoint.pth 文件是保存最新的模型参数的文件。
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            # 如果当前epoch是args.lr_drop的倍数或者是10的倍数，则额外保存一份checkpoint文件
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # 在这里应该不需要把for循环放到if语句下
            # 在训练过程中，通过调用 utils.save_on_master() 函数将训练状态保存到 checkpoint 文件中，所以checkpoint.pth每次都是存放的最新状态的
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 计算模型在验证集上的性能指标。 coco_evaluator 是一个 COCO 格式的评估器，可以用来计算模型在验证集上的 AP 等指标，并将结果以 JSON 格式输出。
        # test_stats 则是一个字典，包含了模型在验证集上的各项性能指标，比如 mAP 等 如果指定了输出目录，则会将 coco_evaluator 对象保存到目录下的 eval.pth 文件中。
        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )
        test_stats_test, coco_evaluator_test = evaluate_test(model, criterion, postprocessors,
                                                             data_loader_test, base_ds_test, device, args.output_dir)
        # 将训练过程中的一些统计数据整合起来，形成一个字典log_stats log_stats包含了训练和测试的各种指标（如loss、accuracy、AP等）
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats_test.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # 将训练过程中的log信息写入到输出目录下的log.txt
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # 单独拿出loss值并存为txt文件
            with (output_dir / "loss.txt").open("a") as f:
                f.write(f"Epoch {epoch}: {log_stats['train_loss']}\n")

            with (output_dir / "loss_test.txt").open("a") as f:
                f.write(f"Epoch {epoch}: {log_stats['test_loss']}\n")

            # for test logs
            if coco_evaluator_test is not None:
                (output_dir /'test').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator_test.coco_eval:
                    with open(results_test_file, "a") as f:
                        # 写入的数据包括coco指标还有loss和learning rate
                        result_test_info = [f"{i:.4f}" for i in coco_evaluator_test.coco_eval["bbox"].stats]
                        txt = "epoch:{} {}".format(epoch, '  '.join(result_test_info))
                        f.write(txt + "\n")
                        ## 打印coco_evaluator_test.coco_eval["bbox"].stats 的内容看看
                    print("打印coco box的内容看看是什么：\n")
                    print(coco_evaluator_test.coco_eval["bbox"].stats)
                    performance = float(result_test_info[1])
                    print("ap50的值是",performance)  # 输出性能指标
                    if performance > best_performance:
                        best_performance = performance
                        # 构建最佳权重文件名，包含 epoch 信息
                        best_checkpoint_path = os.path.join(output_dir/"test",f"checkpoint_best_epoch{epoch}.pth")
                        # 保存当前权重为最佳权重
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, best_checkpoint_path)
                    # 最终只保留一个最佳权重文件，其文件名包含最佳性能对应的 epoch 信息

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



# import schedule
# import time
# print("waiting:!")
# parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()
# schedule.every().day.at("21:28").do(lambda: main(args))  # 每天的10点运行脚本
#
# # 循环执行任务
# while True:
#     schedule.run_pending()
#     time.sleep(1)
