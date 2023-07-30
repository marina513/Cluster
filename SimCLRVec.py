import torch, torchvision, pickle
from torchvision import transforms
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Transformations
batch_size = 8
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
small_288 = transforms.Compose([
                                transforms.Resize((288,512)),
                                transforms.ToTensor(),
                                normalize])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model 
model_path =  "/home/marina/Desktop/Cluster/sscd_disc_mixup.torchscript.pt"
model = torch.jit.load(model_path)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ImgList2VectorSimCLR(ImgsPaths, save_path = None):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dataset & Dataloader
    dataset =  torchvision.datasets.ImageFolder(root = ImgsPaths, transform=small_288)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    paths = [i[0] for i in dataset.imgs]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mean calculation
    embeddings = []
    for batch in tqdm(dataloader):
        imgs, labels = batch
        embedding = model(imgs)
        embedding = embedding.detach().numpy()
        embeddings.extend(embedding)
    mean_vec = np.mean(embeddings, axis=0) 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Vectors calcualtion
    embeddings = embeddings - mean_vec
    embeddings = preprocessing.normalize(embeddings, norm='l2')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save
    if save_path is not None:
        dict_={}
        for i in range(len(paths)):
            dict_[paths[i]] = embeddings[i]

        with open(save_path + 'SimCLR.pkl', 'wb') as f:
            pickle.dump(dict_, f)

    else:
        return embeddings