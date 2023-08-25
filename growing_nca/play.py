import argparse
import torch
import torch.nn as nn
import numpy as np
import pygame
from train import make_seed, to_rgb

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center >= radius
    return mask

def create_erase_mask(size, radius, pos):
    pos = pos * size
    mask = create_circular_mask(size, size, pos, radius)
    return mask
    
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Interact with a neural cellular automata trained model.'
    )
    parser.add_argument(
        'model', 
        type=str,
        help='Path to the model we want to use.'
    )
    # TODO find size, n_channels, and padding from model load
    parser.add_argument(
        '-s', '--size', 
        type=int,
        default=32,
        help='Size of the neural cellular automata grid. Must match the size used during training.'
    )
    parser.add_argument(
        '-r', '--radius', 
        type=int,
        default=4,
        help='Radius of the erase circle when mouse input detected.'
    )
    parser.add_argument(
        '-c', '--n-channels', 
        type=int,
        default=16,
        help='Number of channels in the input tensor. Must match the n_channels used during training.'
    )
    parser.add_argument(
        '-p', '--padding',
        type=int,
        default=16,
        help='Padding. The shape after padding is (h + 2 * p, w + 2 * p).'
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cuda',
        help='Device to use during training.',
        choices=('cpu', 'cuda')
    )
    parser.add_argument(
        '-a', '--scale', 
        type=int,
        default=12,
        help='How much to scale the window. Window size will be (size * scale, size * scale).'
    )
    # parse arguments
    args = parser.parse_args()
    print (vars(args))
    
    # prepare model
    p = args.padding
    device = torch.device(args.device)
    tensor = make_seed(args.size, args.n_channels).to(device)
    tensor = nn.functional.pad(tensor, (p, p, p, p), 'constant', 0)
    
    model = torch.load(args.model)
    model.eval()
    
    pygame.init()
    
    radius = args.radius
    size = args.size + (2 * p)
    scale = args.scale
    window_size = size * scale
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("neural cellular automata")

    running = True
    mouse_down = False
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
            if event.type == pygame.MOUSEMOTION and mouse_down:
                mouse = np.array(pygame.mouse.get_pos(), dtype=float)
                pos = mouse / window_size
                mask = create_erase_mask(size, radius, pos)
                tensor = tensor * torch.tensor(mask).to(device)
            
        # update tensor
        tensor = model(tensor)
        
        # draw tensor to window
        window.fill((255, 255, 255))
        vis = to_rgb(tensor[:, :4].detach().cpu()).squeeze(0).detach().numpy() * 255
        pixel = pygame.Surface((scale, scale))
        
        for j in range(size):
            for i in range(size):
                color = vis[:, i, j]
                pixel.fill(color)
                draw_me = pygame.Rect((j+1)*scale, (i+1)*scale, scale, scale)
                window.blit(pixel, draw_me)
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()