
while(1):
    print("请输入precision:")
    precision = float(input())
    print("请输入recall:")
    recall = float(input())
    f1score = 2*precision*recall/(precision+recall)
    print("f1score的值是:",f1score)
