from dataloader2 import MyDataset2

if __name__ == '__main__':
    
    dataset = MyDataset2('x.csv', 'y.csv')
    x = dataset[0]
    print("HELLO\n\n\n\n\n\n\n\n")
    print(x)