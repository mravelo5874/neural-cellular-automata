import argparse
import torch
import torch.nn as nn
import numpy as np
import pygame
import json
from train import make_seed, to_rgb, create_erase_mask
    
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Interact with a neural cellular automata trained model.'
    )
    parser.add_argument(
        'model', 
        type=str,
        help='Path to the model we want to use.'
    )
    parser.add_argument(
        '-r', '--radius', 
        type=int,
        default=4,
        help='Radius of the erase circle when mouse input detected.'
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
    
    # open params json file
    params = {}
    with open(args.model.replace('.pt', '') + '_params.json', 'r') as openfile:
        # Reading from json file
        params = json.load(openfile)
        
    # prepare model
    radius = args.radius
    p = params['padding']
    device = torch.device(params['device'])
    tensor = make_seed(params['size'], params['n_channels']).to(device)
    tensor = nn.functional.pad(tensor, (p, p, p, p), 'constant', 0)
    model = torch.load(args.model)
    model.eval()

    # prepare pygame instance
    pygame.init()
    size = params['size'] + (2 * p)
    scale = args.scale
    window_size = size * scale
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("neural cellular automata")

    # infinite game loop
    running = True
    mouse_down = False
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
                if pygame.mouse.get_pressed(3)[2]:
                    mouse = np.array(pygame.mouse.get_pos(), dtype=float)
                    pos = mouse / window_size
                    pos = pos * size
                    dot = np.zeros_like(tensor.detach().cpu().numpy())
                    dot[:, :4, int(pos[1]), int(pos[0])] = 255.0
                    tensor += torch.tensor(dot).to(device)
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
            if (event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN) and mouse_down:
                mouse = np.array(pygame.mouse.get_pos(), dtype=float)
                pos = mouse / window_size
                if pygame.mouse.get_pressed(3)[0]:
                    mask = create_erase_mask(size, radius, pos)
                    tensor *= torch.tensor(mask).to(device)
               
            
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
                draw_me = pygame.Rect(j*scale, i*scale, scale, scale)
                window.blit(pixel, draw_me)
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()