import cv2 as cv
import numpy as np
import os
import argparse
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import utilities as utils

def app(path_to_image: str) -> None:
    # wczytanie obrazu
    print('\n__start__\n')
    image_direcotry = path_to_image

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
    samples = utils.sampling(tree_trunk)
    print(f'number of acquired samples: {len(samples)}')
    samples = np.array(samples) # przekształcenie listy na tablicę npumy

    # wczytanie modelu
    model_direcotry = '.\\saved models\\model.keras'
    model = tf.keras.models.load_model(model_direcotry)

    # predict
    predictions = model.predict(samples)

    # prezentacja wyników 
    oak_predisciotn = 0; pine_prediction = 0 
    for prediction in predictions:
        if prediction[0] >= prediction[1]:
            oak_predisciotn += 1
        else:
            pine_prediction += 1
    print(f'{oak_predisciotn}/{len(samples)} samples classified as oak')
    print(f'{pine_prediction}/{len(samples)} samples classified as pine')
    if oak_predisciotn >= pine_prediction:
        print('\nFINAL RESULT: \033[35mOAK\033[0m')
    else:
        print('\nFINAL RESULT: \033[35mPINE\033[0m')


    # wyświetlanie poszczególnych etapów początkowego przetwarzania obrazu
    show_images: bool = False
    if show_images == True:
        while True:
            cv.imshow('Input Image', tree_image)
            cv.imshow('Cut out tree trunk', tree_trunk)
            
            key = cv.waitKey(0)
            if key == ord('q') or key == 27:
                cv.destroyAllWindows()
                break

    print('\n__end__\n')

def file_path(path: str) -> str:
    if os.path.isfile(path):
        if path.split('.')[-1] == 'jpg':
            return path
    raise argparse.ArgumentError(f'Invalid file: {path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=file_path)
    args = parser.parse_args()
    path_to_image = args.path

    app(path_to_image)    

if __name__ == '__main__':
    main()