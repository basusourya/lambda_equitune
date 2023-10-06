import torch
import clip
from tqdm import tqdm

model_name_dict = {
'RN50': "RN50",
'RN101': "RN101",
'RN50x4': "RN50x4",
'RN50x16': "RN50x16",
'RN50x64': "RN50x64",
'ViT-B/32': "ViT_B_32",
'ViT-B/16': "ViT_B_16",
'ViT-L/14': "ViT_L_14",
'ViT-L/14@336px': "ViT_L_14_336px"
}


def zeroshot_classifier(args, model, classnames, templates, save_weights='True'):
    import os
    dir_path = os.path.join("saved_zeroshot_weights")  # dir to save the trained models

    model_name_to_save = model_name_dict[args.model_name]
    file_name = os.path.join(args.dataset_name + model_name_to_save + ".pt")  # filename for the trained models

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, file_name)


    if os.path.exists(file_path):
        zeroshot_weights = torch.load(file_path)
        print(f"loaded zeroshot weights!")
    else:
        print(f"computing zeroshot weights...")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)  # average over all 80 prompts
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)  # each class has a weight of size 512
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

        # save weights
        torch.save(zeroshot_weights, file_path)
    return zeroshot_weights