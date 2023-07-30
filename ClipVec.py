import clip
from PIL import Image
import numpy as np
import torch, pickle

# load Model
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()


# Main Function
def ImgList2Vector(ImgsPaths, save_path = None):

    # paths to PIL img
    images = []
    for filename in ImgsPaths:
        image = Image.open(filename).convert("RGB")
        images.append(preprocess(image))
    
    # PIL img to tensors 
    image_input = torch.tensor(np.stack(images)).cuda()

    # tensors to features
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    
    # normalize
    
    image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)

    if save_path is not None:
        dict_={}
        for i in range(len(ImgsPaths)):
            dict_[ImgsPaths[i]] = image_features_normalized[i]

        with open(save_path , 'wb') as f:
            pickle.dump(dict_, f)

    else:
        return image_features, image_features_normalized