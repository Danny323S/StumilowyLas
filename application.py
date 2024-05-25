import cv2 as cv
import numpy as np
import os
# import tensorflow as tf

import utilities as utils

def main():
    # wczytanie obrazu
    print('\n__start__')
    image_direcotry = '.\\drzewka\\sosna_1.jpg'

    if not os.path.isfile(image_direcotry):
        print(f'File {image_direcotry} does not exist.')
        exit()
 
    tree_image = cv.imread(image_direcotry)
    print(f'Oryginal image resolution : {tree_image.shape}')
    # zmniejszenie obrazu jeżeli to konieczne
    while tree_image.shape[0] > 1500:
        tree_image = cv.pyrDown(tree_image)
    print(f'Rescaled image resolution : {tree_image.shape}')

    # wycinanie pnia drzewa ze zdjęcia
    tree_trunk_mask =  utils.find_tree_trunk(tree_image)
    cropped_mask = utils.crop_mask(tree_trunk_mask)
    tree_trunk = cv.bitwise_and(tree_image, tree_image, mask=cropped_mask)
    tree_trunk = tree_trunk[:, min(set(np.where(tree_trunk != 0)[1])):max(set(np.where(tree_trunk != 0)[1]))]
    
    # sampling

    # wczytanie modelu

    # predict

    # prezentacja wyników 

    # wyświetlanie poszczególnych etapów początkowego przetwarzania obrazu
    show_images: bool = True
    if show_images == True:
        while True:
            cv.imshow('Input Image', tree_image)
            cv.imshow('Cut out tree trunk', tree_trunk)
            
            key = cv.waitKey(0)
            if key == ord('q') or key == 27:
                cv.destroyAllWindows()
                break

    print('__end__\n')

if __name__ == '__main__':
    main()