import argparse
import torch
import torch.nn as nn
import numpy as np
import pygame
import json
import datetime
import os

from models import NCA_grow_learnable, NCA_grow_laplace, NCA_grow_sobel
from utils import Utils

def load_model(_model, _dir, _device):
    print ('loading model: ' + _model)
    # open params json file
    params = {}
    with open(_dir +'\\'+ _model + '_params.json', 'r') as openfile:
        # Reading from json file
        params = json.load(openfile)
    
    if params['model_type'] == 'learnable':
        model = NCA_grow_learnable()
    elif params['model_type'] == 'laplace':
        model = NCA_grow_laplace()
    elif params['model_type']  == 'sobel':
        model = NCA_grow_sobel()
    model.load_state_dict(torch.load(_dir +'\\'+_model + '.pt', map_location=_device))
    model.eval()
    
    p = params['pad']
    tensor = Utils.make_seed(params['size'], params['n_channels']).to(_device)
    tensor = nn.functional.pad(tensor, (p, p, p, p), 'constant', 0)
    
    return model, tensor, params
    
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Interact with a neural cellular automata trained model.'
    )
    parser.add_argument(
        '-m', '--models',
        type=str,
        default='models',
        help='Path to the model directory we want to use.'
    )
    parser.add_argument(
        '-r', '--radius', 
        type=int,
        default=8,
        help='Radius of the erase circle when mouse input detected.'
    )
    parser.add_argument(
        '-s', '--scale', 
        type=int,
        default=10,
        help='How much to scale the window. Window size will be (size * scale, size * scale).'
    )
    # * parse arguments
    args = parser.parse_args()
    print ('args: ', vars(args))
    
    # * get list of models
    model_list = os.listdir(args.models)
    model_list = [item.replace('.pt', '') for item in model_list if item.endswith('.pt')]
    print ('models: ', model_list)
    curr = 0
    
    # * load current model
    device = torch.device('cpu')
    model, tensor, params = load_model(model_list[curr], args.models, device)

    # * misc params
    angle = 0.0
    fps = 0
    radius = args.radius
    prev_time = datetime.datetime.now()
    
    # * start pygame
    pygame.init()
    pygame.display.set_caption('nca play - ' + model_list[curr])
    
    # * model dependent params
    p = params['pad']
    size = params['size'] + (2 * p)
    scale = args.scale
    window_size = size * scale
    window = pygame.display.set_mode((window_size, window_size))
    
    # * text renders
    font_size = 24
    font_color = (255, 255, 255)
    my_font = pygame.font.SysFont('consolas', font_size)
    model_surface = my_font.render('model: ' + model_list[curr], False, font_color)
    text_surface = my_font.render('angle: ' + str(angle) + 'π', False, font_color)
    fps_surface = my_font.render('fps: ' + str(int(fps)), False, font_color)

    # * start infinite game loop
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
                    tensor = Utils.make_seed(params['size'], params['n_channels']).to(device)
                    tensor = nn.functional.pad(tensor, (p, p, p, p), 'constant', 0)
                if event.key == pygame.K_UP:
                    curr += 1
                    if curr >= len(model_list):
                        curr = 0
                if event.key == pygame.K_DOWN:
                    curr -= 1
                    if curr < 0:
                        curr = len(model_list)-1
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    # load new model
                    model, tensor, params = load_model(model_list[curr], args.models, device)
                    p = params['pad']
                    pygame.display.set_caption('nca play - ' + model_list[curr])
                    size = params['size'] + (2 * p)
                    scale = args.scale
                    window_size = size * scale
                    window = pygame.display.set_mode((window_size, window_size))
                    model_surface = my_font.render('model: ' + model_list[curr], False, font_color)
            if event.type == pygame.MOUSEWHEEL:
                angle = np.round((event.y * 0.1) + angle, decimals=1)
                text_surface = my_font.render('angle: ' + str(angle) + 'π', False, font_color)
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
                    mask = Utils.create_erase_mask(size, radius, pos)
                    tensor *= torch.tensor(mask).to(device)
               
        # * update tensor
        with torch.no_grad():
            tensor = model(tensor, angle * np.pi)
        
        # * draw tensor to window
        window.fill((255, 255, 255))
        vis = Utils.to_rgb(tensor[:, :4].detach().cpu()).squeeze(0).detach().numpy() * 255
        pixel = pygame.Surface((scale, scale))
        for j in range(size):
            for i in range(size):
                color = vis[:, i, j]
                pixel.fill(color)
                draw_me = pygame.Rect(j*scale, i*scale, scale, scale)
                window.blit(pixel, draw_me)
        
        # * calculate fps
        now = datetime.datetime.now()
        if (now - prev_time).seconds >= 1.0:
            prev_time = now
            fps_surface = my_font.render('fps: ' + str(int(fps)), False, font_color)
            fps = 0
        else:
            fps += 1       
        
        # * render text
        window.blit(model_surface, (0, 0))
        window.blit(text_surface, (window_size-font_size-150, window_size-font_size))
        window.blit(fps_surface, (0, window_size-font_size))
        
        # * flip it!
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()