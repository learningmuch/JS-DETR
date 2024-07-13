import matplotlib.pyplot as plt

# 读取txt文件
# 这几句代码是打开名为'results20230417-191652.txt'的文本文件并读取文件中的每一行，使用strip()函数去除每行开头和结尾的空格和换行符，并用split(' ')函数将每行内容以空格为分隔符拆分为列表形式。
#
# 然后对于每行拆分出来的列表，取出其中第一个元素并将其转换为int类型，取出其中第四个元素并将其转换为float类型，将这两个数据作为一组数据(x, y)添加到data列表中。
#
# 因为每行的第一个元素类似于'epoch:9'的格式，所以使用split(':')函数将其以':'为分隔符拆分为列表形式，取出其中的第二个元素并将其转换为int类型，就可以得到epoch的数值。而每行的第四个元素就是我们需要绘制的数据，因此直接将其转换为float类型即可。
# data = []
# with open(r'E:\DETR\detr.6.8\weight_all\DA\test20230613-211001.txt', 'r') as f:
#     for line in f:
#         items = line.strip().split(' ')
#         data.append((int(items[0].split(':')[1]), float(items[3])))
#
# # 绘制折线图
# x = [d[0] for d in data]
# y = [d[1] for d in data]
# plt.plot(x, y)
# plt.xlabel('epoch')
# plt.ylabel('map')
# plt.title('loss vs ap')
# plt.show()
# plt.savefig('sa_loss.png')



# # 读取数据
data = []
# with open('/home/zcs/code/pycode/deep-learning-for-image-processing-master/pytorch_object_detection/detr-main/weights_sa1/loss.txt', 'r') as f:
#     for line in f:
#         epoch, loss = line.strip().split(': ')
#         data.append((int(epoch.split(' ')[-1]), float(loss)))

with open(r'E:\DETR\detr.6.8\weight_all\DA\loss.txt', 'r') as f:
    for line in f:
        epoch, loss = line.strip().split(': ')
        data.append((int(epoch.split(' ')[-1]), float(loss)))

# 绘制折线图
x = [d[0] for d in data]
y = [d[1] for d in data]
plt.plot(x, y)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
plt.savefig('sa_loss.png')

#
#

