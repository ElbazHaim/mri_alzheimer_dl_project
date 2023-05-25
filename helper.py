"""
Helper module for the CNN MRI-Alzheimer's Stage project, contains the utility code.
"""
import matplotlib.pyplot as plt
from torchvision import transforms



def show_images(imgs: list, labels: list, idx_to_class: dict) -> None:
    """
    Function to plot images from dataloader in a row, with label names as titles.
    
    imgs: list: Image tensors
    labels: list: Label tensor (dataloader output)
    idx_to_class: dict: A dictionary to transform classes to their names
    
    return: None
    """
    if not isinstance(imgs, list):
        imgs = [imgs] 
    labels = labels.numpy()
    
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(img)
        label = idx_to_class[labels[i]]
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=label)
    fig.tight_layout()
    

    

if __name__ == "__main__":
    raise ImportError("helper module executed as main")
