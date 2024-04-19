import numpy as np
import cv2 as cv
import os

def getImagePath(folder_directory):
    '''funkcja zwracająca listę ścieżek do obrazów ".jpg" w podanym folderze.'''
    images_path = []
    for images in os.listdir(folder_directory):
        if images.endswith('.jpg'):
            images_path.append(folder_directory + '\\' + images)
    return images_path

def main ():
    #ustalenie ścieżki do folderu z obrazami:
    paths_to_images = getImagePath('resources\\drzewa\\dab')
    # paths_to_images = load_image('resources\\drzewa\\sosna')

    #wczytywanie kolejnych obrazów
    for im_path in paths_to_images:
        image = cv.imread(im_path)
        image = cv.pyrDown(image)
        print(f'Acctual file: {im_path.split('\\')[-1]}')
        cv.imshow('Oryginal Image', image)

        key = cv.waitKey(0)
        if key == ord('q') or key == 27:
            exit()

if __name__ == '__main__':
    main()