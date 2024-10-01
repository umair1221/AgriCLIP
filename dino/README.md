
# Using Self-Supervised Model (DINO)

## Training

### Documentation
Please install [PyTorch](https://pytorch.org/) and download the [ALive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/umair_nawaz_mbzuai_ac_ae/Ev3ZGFeLw8JPkda2RcRv_e0BLXqi20bFRhA2kISEwEQSXw?e=1earRb) dataset. This codebase is being used from the [DINO](https://github.com/facebookresearch/dino) repository and we are thankful for providing well-explained documentation. To train DINO model, you'll require python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. 

### DINO training :sauropod:
Run DINO with ViT-base network on a single node with 1 GPU for 100 epochs with the following command. 

```
python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path /path/to/ALive/SelfSupervised/data --output_dir /path/to/output/directory
```

## Citation
We are thankful and would like to give credit to the following paper:
```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
