from dataloader import MyDataset2
from model2 import SentimentModel
from sentiment import GloveLayer
import os

if __name__ == '__main__':
    dataset = MyDataset2('../../cleaned_data', 'x.csv', 'y.csv')
    home, homenews, away, awaynews, y = dataset[2:4]
    #print("HELLO\n\n\n\n\n\n\n\n")
    #print(homenews)
    #print(homenews.shape)

    hidden_size = 256
    feature_size = 50
    glove_path = os.path.join(os.getcwd(), 'glove.6B.50d.txt')

    model = SentimentModel(feature_size, hidden_size, glove_path)

    outa, outb = model(homenews, awaynews)

    print(outa.shape)
    print(outa)