import argparse
import torch
import torch.nn as nn
import numpy as np
import pygame
import json

from model import NCA_model
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
        default=8,
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
    angle = 0.0
    radius = args.radius
    p = params['pad']
    
    device = torch.device('cpu')
    model = NCA_model()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tensor = make_seed(params['size'], params['n_channels']).to(device)
    tensor = nn.functional.pad(tensor, (p, p, p, p), 'constant', 0)

    # prepare pygame instance
    pygame.init()
    size = params['size'] + (2 * p)
    scale = args.scale
    window_size = size * scale
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption('nca play - ' + params['name'])
    
    my_font = pygame.font.SysFont('consolas', 16)
    text_surface = my_font.render('angle: ' + str(angle), False, (0, 0, 0))

    # infinite game loop
    running = True
    mouse_down = False
    while running:
        # empty cache
        torch.cuda.empty_cache()
        
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    tensor = make_seed(params['size'], params['n_channels']).to(device)
                    tensor = nn.functional.pad(tensor, (p, p, p, p), 'constant', 0)
            if event.type == pygame.MOUSEWHEEL:
                angle += event.y * 0.5
                text_surface = my_font.render('angle: ' + str(angle), False, (0, 0, 0))
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
                if pygame.mouse.get_pressed(3)[2]:
                    mouse = np.array(pygame.mouse.get_pos(), dtype=float)
                    pos = mouse / window_size
                    pos = pos * size
                    dot = np.zeros_like(tensor.detach().cpu().numpy())
                    dot[:, 3:, int(pos[1]), int(pos[0])] = 1.0
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
        with torch.no_grad():
            tensor = model(tensor, angle)
        
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
                
        # show text
        window.blit(text_surface, (0,0))
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()