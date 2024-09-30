import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from TextToConcept import TextToConcept
# Import the dictionary from prompt.py
from AgriCLIP_alignment.prompts import prompts_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Script for zero-shot classification using a trained aligner.')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Name of the dataset which selects the prompt template')
    parser.add_argument('--data-path', type=str, default='/path/to/dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--dino-path', type=str, default='/path/to/dino.pth',
                        help='Path to DINO model')
    parser.add_argument('--aligner-path', type=str, default='/path/to/aligner.pth',
                        help='Path to the aligner model')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for the DataLoader')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for the DataLoader')
    parser.add_argument('--prompt-template', type=str, default=None,
                        help='Template for generating prompts for zero-shot classification')
    return parser.parse_args()

def main():
    args = parse_args()

    # Select the prompt based on dataset name or use the provided prompt template
    if args.prompt_template is None and args.dataset_name:
        args.prompt_template = prompts_dict.get(args.dataset_name, 'a photo of {}')  # Default prompt if not found

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    dataset = ImageFolder(root=args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize TextToConcept with the model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=True)
    model.load_state_dict(torch.load(args.dino_path , weights_only=True))
    def forward_features(x):
        # Assuming the model returns the last layer's output which might include a classification token or similar
        features = model(x)  # This call should be adjusted based on the actual output format
        # You might need to process the features here, e.g., selecting the right tensor or applying global pooling
        return features.squeeze()  # Adjust this based on the actual structure of your features


    # Attach the custom forward method to the model
    model.forward_features = forward_features

    model.to(device)
    text_to_concept = TextToConcept(model, 'Dino')
    text_to_concept.load_linear_aligner(args.aligner_path)

    # Class mapping
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names_list = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    print(class_names_list)

    # Zero-shot classifier
    cifar_zeroshot_classifier = text_to_concept.get_zero_shot_classifier(
        class_names_list, prompts=[args.prompt_template])

    # Zero-shot classification
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(loader):
            x, y = data[:2]
            x = x.to(device)
            try:
                outputs = cifar_zeroshot_classifier(x).detach().cpu()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            except IndexError as e:
                print(f"Error processing batch: {e}")
                continue

    print(f'Zeroshot Accuracy: {100.*correct/total:.2f}%')

if __name__ == '__main__':
    main()
