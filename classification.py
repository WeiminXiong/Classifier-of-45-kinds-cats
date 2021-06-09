import classifier
import torch
import cv2
import os



def classify(img):
    '''
    :param img: 输入的图像，大小为224*224
    :return: probs :top5的各种类的概率
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("cuda available:", torch.cuda.is_available())
    print("current device:",device)

    idx2breed={0: 'Abyssinian', 1: 'American Bobtail', 2: 'American Curl',
               3: 'American Shorthair', 4: 'American Wirehair', 5: 'Balinese',
               6: 'Bengal', 7: 'Birman', 8: 'Bombay', 9: 'British Shorthair',
               10: 'Burmese', 11: 'Burmilla', 12: 'Chartreux',
               13: 'Colorpoint Shorthair', 14: 'Cornish Rex', 15: 'Devon Rex',
               16: 'Egyptian Mau', 17: 'European Burmese', 18: 'Exotic',
               19: 'Havana Brown', 20: 'Japanese Bobtail', 21: 'Khao Manee',
               22: 'Korat', 23: 'LaPerm', 24: 'Lykoi',
               25: 'Maine Coon Cat', 26: 'Manx', 27: 'Norwegian Forest Cat', 28: 'Ocicat',
               29: 'Oriental', 30: 'Persian', 31: 'RagaMuffin', 32: 'Ragdoll',
               33: 'Russian Blue', 34: 'Scottish Fold', 35: 'Selkirk Rex',
               36: 'Siamese', 37: 'Siberian', 38: 'Singapura', 39: 'Somali',
               40: 'Sphynx', 41: 'Tonkinese', 42: 'Toybob', 43: 'Turkish Angora', 44: 'Turkish Van'}


    models = []
    model = classifier.pretrained_resnet152()
    model.load_state_dict(torch.load('./models/resnet152_filtered.pth'))
    models.append(model.to(device))
    model = classifier.pretrained_resnet101()
    model.load_state_dict(torch.load('./models/resnet101_filtered.pth'))
    models.append(model.to(device))
    model = classifier.pretrained_resnet50()
    model.load_state_dict(torch.load('./models/resnet50_filtered.pth'))
    models.append(model.to(device))
    model = classifier.pretrained_resnet34()
    model.load_state_dict(torch.load('./models/resnet34_filtered.pth'))
    models.append(model.to(device))
    for model in models:
        model.eval()

    img=torch.Tensor(img).to(device)
    img=img.permute(2,0,1)
    img=img.unsqueeze(dim=0)
    print(img.shape)
    pred_score=torch.zeros((1,45),device=device)
    for model in models:
        pred_score+=model(img)
    pred_score/=len(models)
    pred_probs=torch.softmax(pred_score,dim=1)
    res,ind=torch.sort(pred_probs,descending=True)
    probs=[]
    for i in range(5):
        probs.append((idx2breed[ind[0][i].item()],res[0][i].item()))
    return probs


if __name__=="__main__":
    filename='./filtered_data/test/American_Curl/8.jpg'
    img=cv2.imread(filename)
    img=cv2.resize(img,(224,224))
    probs=classify(img)
    print(probs)