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

def findTreeTrunk(src_image):
    '''funkcja zwracająca obszar na którym znajduje się pień drzewa'''
    image = src_image
    #tworzenie maski dla metody grabCut 
    mask = np.zeros(list(src_image.shape)[:-1], np.uint8)

    #oznaczenie pionową linią - pewnego pierwszego planu
    start_point =(src_image.shape[1]//2, 0)
    end_point = (src_image.shape[1]//2, src_image.shape[0])
    fg_color = 255
    thickness = 5
    cv.line(mask, start_point, end_point, fg_color, thickness)

    #oznaczenie pionową linią - pewnego tła
    thickness = 10
    start_point_left = (0 + thickness//2, 0)
    end_point_left = (0 + thickness//2, src_image.shape[0])
    start_point_right = (src_image.shape[1] - (thickness//2), 0)
    end_point_right= (src_image.shape[1] - (thickness//2), src_image.shape[0])
    bg_color = 20
    cv.line(mask, start_point_left, end_point_left, bg_color, thickness)
    cv.line(mask, start_point_right, end_point_right, bg_color, thickness)

    mask[mask == fg_color] = cv.GC_FGD #pixele, które na pewno zawierają pierwszy plan
    mask[mask == bg_color] = cv.GC_BGD #pixele, które na pewno zawierają tło
    mask[mask == 0] = cv.GC_PR_BGD #pixele, które mogą zawierać tło

    #zastosowanie metody grabCut
    fg_model = np.zeros((1, 65), np.float64) #model pierwszego planu -> wymagane dla metody
    bg_model = np.zeros((1, 65), np.float64) #model tła -> wymagane dla metody
    iterations = 30 #liczba iteracji algorytmu grab cat (Ten parametr trzaba lepiej określić)
    mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model,
        iterations, mode=cv.GC_INIT_WITH_MASK)
    
    #wizualizacjia otrzymanej maski
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask

def main ():
    #ustalenie ścieżki do folderu z obrazami:
    paths_to_images = getImagePath('resources\\drzewa\\dab')
    # paths_to_images = getImagePath('resources\\drzewa\\sosna')

    #wczytywanie kolejnych obrazów
    for im_path in paths_to_images:
        image = cv.imread(im_path)
        image = cv.pyrDown(image)
        #wyświetlenie na konsolę postępu
        print(f'Progress: {paths_to_images.index(im_path)+1}/{len(paths_to_images)}')
        print(f'Acctual file: {im_path.split('\\')[-1]}')
        mask = findTreeTrunk(image) #maska
        #zastosowanie maski do oryginalnego obrazu
        tree_trunk_image = image*mask[:,:,np.newaxis] 

        save_output = True
        if save_output == True:
            output_file_name = 'masked-' + im_path.split('\\')[-1]
            output_diectory = ('output_fils\\tree_trunks\\' +
                im_path.split('\\')[-2]) + '\\'
            cv.imwrite(output_diectory+output_file_name, tree_trunk_image)
        
        #wyświetlenie obrazów,     
        show_results = False
        if show_results == True:
            cv.imshow('Oryginal Image', image)
            cv.imshow('Mask', np.where(mask == 1, 255, 0).astype('uint8'))
            cv.imshow('Tree trunk', tree_trunk_image)

            key = cv.waitKey(0)
            if key == ord('Q') or key == ord('q') or key == 27:
                exit()

if __name__ == '__main__':
    main()