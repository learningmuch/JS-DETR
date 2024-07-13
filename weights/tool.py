
# # 通过载入这个文件并读取里面的内容来获取其保存的epoch
# import torch
#
# # 加载checkpoint文件
# checkpoint = torch.load("checkpoint.pth")
#
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
# # 获取epoch
# epoch = checkpoint['epoch']
# 获取optimizer状态字典
# checkpoint = torch.load(checkpoint_path)
# optimizer_state_dict = checkpoint['optimizer']
# print(optimizer_state_dict)






# 可以使用以下代码读取 eval.pth 并输出每个类别的平均精度（AP）

# import torch
#
# eval_file = 'eval.pth'
# eval_data = torch.load(eval_file)
#
# # 输出每个类别的平均精度（AP）
# for i, category in enumerate(eval_data['bbox'].categories):
#     print(f'AP for {category}: {eval_data["bbox"].stats[i]}')


#

# 将指标保存到一个csv文件中，每一行对应一个epoch的指标。output_dir 是存放路径
# import csv
#
# # 在训练开始前创建csv文件
# with open(output_dir / "stats.csv", mode="w") as f:
#     writer = csv.writer(f)
#     writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

# import csv
#
# # 在每个epoch结束后将指标保存到csv文件中
# with open(output_dir / "stats.csv", mode="a") as f:
#     writer = csv.writer(f)
#     writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

