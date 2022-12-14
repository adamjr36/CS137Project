from dataloader import MyDataset2

if __name__ == '__main__':


    
    dataset = MyDataset2('../../cleaned_data', 'x.csv', 'y.csv')
    home, homenews, away, awaynews, y = dataset[2:4]
    print("HELLO\n\n\n\n\n\n\n\n")
    print(homenews)
    print(homenews.shape)