import argparse
import pathlib
import datetime
import random
import numpy as np
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
from model import NCA_model

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center >= radius
    return mask

def create_erase_mask(size, radius, pos):
    pos = pos * size
    mask = create_circular_mask(size, size, pos, radius)
    return mask

def load_image(path, size):
    """
    Loads an image from a specified path and converts to torch.Tensor
    
    Parameters
    ----------
    path : pathlib.Path
        Path to where the image is located. None that the image needs to be RGBA.
        
    size : int
        The image will be resized to a square with a length of 'size'.
        
    Returns
    -------
    torch.Tensor
        4D float image of shape `(1, 4, size, size)`. The RGB channels are premultiplied
        by the alpha channel.
    """
    img = Image.open(path)
    img = img.resize((size, size), Image.LANCZOS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]
    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]

def to_rgb(img_rgba):
    """
    Converts an RGBA image to a RGB image.
    
    Parameters
    ----------
    img_rgba : torch.Tensor
        4D tensor of shape `(1, 4, size, size)` where the RGB channels were already
        premultiplied by the alpha.
        
    Returns
    -------
    img_rgb : torch.Tensor
          4D tensor of shape `(1, 3, size, size)`.
    """
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)

def make_seed(size, n_channels):
    """
    Create a starting tensor for training.
    Only the active pixels are goin to be in the middle.
    
    Parameters
    ----------
    size : int
        The height and the width of the tensor.
    
    n_channels : int 
        Overall number of channels. Note that it needs to be higher than 4 since the 
        first 4 channels represent RGBA.
        
    Returns
    -------
    torch.Tensor
        4D tensor of shape `(1, n_channels, size, size)`.
    """
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Training script for neural cellular automata.'
    )
    parser.add_argument(
        'img', 
        type=str,
        help='Path to the image we want to reproduce.'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=8,
        help='Batch size. Samples will always be taken randomly from the pool.'
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cuda',
        help='Device to use during training.',
        choices=('cpu', 'cuda')
    )
    parser.add_argument(
        '-e', '--eval-frequency',
        type=int,
        default=500,
        help='Evaluation frequency.'
    )
    parser.add_argument(
        '-i', '--eval-iterations',
        type=int,
        default=300,
        help='Number of iterations when evaluating.'
    )
    parser.add_argument(
        '-n', '--n-batches',
        type=int,
        default=10000,
        help='Number of batches to train for.'
    )
    parser.add_argument(
        '-c', '--n-channels',
        type=int,
        default=16,
        help='Number of channels in the input tensor.'
    )
    parser.add_argument(
        '-l', '--logdir',
        type=str,
        default='logs',
        help='Folder whenre all the logs and outputs are saved.'
    )
    parser.add_argument(
        '-p', '--padding',
        type=int,
        default=16,
        help='Padding. The shape after padding is (h + 2 * p, w + 2 * p).'
    )
    parser.add_argument(
        '--pool-size',
        type=int,
        default=1024,
        help='Size of the training pool.'
    )
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=32,
        help='Image size.'
    )
    parser.add_argument(
        '-save', "--save-model",
        type=bool,
        default=True,
        help='Save the model after training as .pt file.'
    )
    parser.add_argument(
        '-m', '--modeldir',
        type=str,
        default='models',
        help='Where to save the model after training.'
    )
    parser.add_argument(
        '-name', '--name',
        type=str,
        default=None,
        help='Where to save the model after training.'
    )
    parser.add_argument(
        '-dam', '--damage',
        type=int,
        default=3,
        help='Number of examples in a batch to damage.'
    )
    # parse arguments
    args = parser.parse_args()
    print (vars(args))
    print('cuda available:', torch.cuda.is_available())
    
    # misc
    p = args.padding
    full_size = args.size + (2 * p)
    device = torch.device(args.device)
    
    # create log
    log_path = pathlib.Path(args.logdir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    # target image
    target_img_ = load_image(args.img, size=args.size)
    target_img_ = nn.functional.pad(target_img_, (p, p, p, p), 'constant', 0)
    target_img = target_img_.to(device)
    target_img = target_img.repeat(args.batch_size, 1, 1, 1)
    writer.add_image('ground truth', to_rgb(target_img_)[0])
    
    # model and optimizer
    model = NCA_model(_n_channels=args.n_channels, _device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    # pool init
    seed = make_seed(args.size, args.n_channels).to(device)
    seed = nn.functional.pad(seed, (p, p, p, p), 'constant', 0)
    pool = seed.clone().repeat(args.pool_size, 1, 1, 1)
    
    # training loop
    for it in tqdm(range(args.n_batches)):
        batch_ixs = np.random.choice(
            args.pool_size, args.batch_size, replace=False
        ).tolist()
        
        # get training batch
        x = pool[batch_ixs]
        
        # damage examples in batch
        if args.damage > 0:
            radius = random.uniform(args.size*0.1, args.size*0.4)
            u = random.uniform(0, 1) * args.size + p
            v = random.uniform(0, 1) * args.size + p
            mask = create_erase_mask(full_size, radius, [u, v])
            x[-args.damage:] *= torch.tensor(mask).to(device)
        
        # forward pass
        for i in range(np.random.randint(64, 96)):
            x = model(x)
        
        loss_batch = ((target_img - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
        loss = loss_batch.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss, it)
        
        # find best in batch
        argmax_batch = loss_batch.argmax().item()
        argmax_pool = batch_ixs[argmax_batch]
        remaining_batch = [i for i in range(args.batch_size) if i != argmax_batch]
        remaining_pool = [i for i in batch_ixs if i != argmax_pool]
        
        pool[argmax_pool] = seed.clone()
        pool[remaining_pool] = x[remaining_batch].detach()
        
        if it % args.eval_frequency == 0:
            x_eval = seed.clone()
            eval_video = torch.empty(1, args.eval_iterations, 3, *x_eval.shape[2:])
            for it_eval in range(args.eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out
                
            writer.add_video('eval', eval_video, it, fps=60)
            
    # save model
    if args.save_model:
        model_path = pathlib.Path(args.modeldir)
        model_path.mkdir(parents=True, exist_ok=True)
        if args.name == None:
            ts = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
            args.name = 'model_' + ts
        torch.save(model, args.modeldir + '\\' + args.name + '.pt')
        
        # save model arguments
        dict = vars(args)
        json_object = json.dumps(dict, indent=4)
        with open(args.modeldir + '\\' + args.name + '_params.json', 'w') as outfile:
            outfile.write(json_object)

if __name__ == "__main__":
    main()