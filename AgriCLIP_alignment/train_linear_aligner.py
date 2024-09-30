import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from tqdm import tqdm
from TextToConcept import TextToConcept

def parse_args():
    parser = argparse.ArgumentParser(description='Script for training and saving an aligned model.')
    parser.add_argument('--data-path', type=str, default='/home/umair.nawaz/Research_Work/Submission/AgriClip/Pre-Train',
                        help='Path to the dataset directory')
    parser.add_argument('--dino-weights-path', type=str, 
                        default='/home/umair.nawaz/Research_Work/Submission/AgriClip/Weights/dino_pretrain.pth',
                        help='Path to the pre-trained DINO model weights')
    parser.add_argument('--path-dino-features', type=str, 
                        default='/home/umair.nawaz/Research_Work/Submission/AgriClip/Aligned_Models/DINO/DINO.npy',
                        help='Path to DINO features')
    parser.add_argument('--path-clip-features', type=str, 
                        default='/home/umair.nawaz/Research_Work/Submission/AgriClip/Aligned_Models/CLIP/CLIP.npy',
                        help='Path to CLIP features')
    parser.add_argument('--output-model-path', type=str, default='/home/umair.nawaz/Research_Work/Submission/AgriClip/Aligned_Models/AgriCLIP_Aligned.pth',
                        help='Path to save the aligned model')
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=True)
    model.load_state_dict(torch.load(args.dino_weights_path, map_location=device, weights_only=True))
    model.to(device)

    # Transformation setup
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Dataset setup
    dset = ImageFolder(root=args.data_path, transform=transform)
    num_of_samples = int(0.99 * len(dset))
    dset = Subset(dset, np.random.choice(np.arange(len(dset)), num_of_samples, replace=False))

    # TextToConcept initialization
    text_to_concept = TextToConcept(model, 'Dino')
    text_to_concept.train_linear_aligner(dset,
                                         save_reps=False, load_reps=True,
                                         path_to_model=args.path_dino_features,
                                         path_to_clip_model=args.path_clip_features,
                                         epochs=70)

    # Saving the model
    text_to_concept.save_linear_aligner(args.output_model_path)
    print("Aligned model saved successfully...")

if __name__ == '__main__':
    main()
