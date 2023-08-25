import argparse
import torch
import pygame

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
        '-s', '--size', 
        type=int,
        default=32,
        help='Size of the neural cellular automata grid. Must match the size used during training.'
    )
    parser.add_argument(
        '-c', '--scale', 
        type=int,
        default=4,
        help='How much to scale the window. Window size will be (size * scale, size * scale).'
    )
    # parse arguments
    args = parser.parse_args()
    print (vars(args))
    
    model = torch.load(args.model)
    model.eval()
    
    pygame.init()
    
    side = args.size * args.scale 
    window = pygame.display.set_mode((side, side))
    pygame.display.set_caption("neural cellular automata")

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        # Draw shapes
        window.fill((255, 255, 255))
        pygame.draw.circle(window, (255, 0, 0), (150, 200), 50)
        pygame.draw.rect(window, (0, 200, 0), (100, 300, 300, 200))
        pygame.draw.line(window, (0, 0, 100), (100, 100), (700, 500), 5)
        
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()