import cv2 as cv
import numpy as np
import os
from cv2 import UMat
def find_tree_trunk(src_image: UMat) -> UMat: 
    '''funkcja zwracająca obszar na którym znajduje się pień drzewa'''
    image = src_image
    #tworzenie maski dla metody grabCut 
    mask = np.zeros(list(src_image.shape)[:-1], np.uint8)

    #oznaczenie pionową linią - pewnego pierwszego planu
    start_point =(src_image.shape[1]//2, 0)
    end_point = (src_image.shape[1]//2, src_image.shape[0])
    fg_color = 255
    fg_thickness = int(0.02*src_image.shape[0])
    cv.line(mask, start_point, end_point, fg_color, fg_thickness)

    #oznaczenie pionową linią - pewnego tła
    bg_thickness = int(0.03*src_image.shape[0])
    start_point_left = (0 + bg_thickness//2, 0)
    end_point_left = (0 + bg_thickness//2, src_image.shape[0])
    start_point_right = (src_image.shape[1] - (bg_thickness//2), 0)
    end_point_right= (src_image.shape[1] - (bg_thickness//2), src_image.shape[0])
    bg_color = 20
    cv.line(mask, start_point_left, end_point_left, bg_color, bg_thickness)
    cv.line(mask, start_point_right, end_point_right, bg_color, bg_thickness)

    mask[mask == fg_color] = cv.GC_FGD #pixele, które na pewno zawierają pierwszy plan
    mask[mask == bg_color] = cv.GC_BGD #pixele, które na pewno zawierają tło
    mask[mask == 0] = cv.GC_PR_BGD #pixele, które mogą zawierać tło

    show_grabCut_imput = False
    if show_grabCut_imput == True:
        rgb_mask = np.zeros((src_image.shape[0], src_image.shape[1], 3), np.uint8)
        cv.line(rgb_mask, start_point, end_point, (0,0,255), fg_thickness)
        cv.line(rgb_mask, start_point_left, end_point_left, (0,255,0), bg_thickness)
        cv.line(rgb_mask, start_point_right, end_point_right, (0,255,0), bg_thickness)
        cv.imshow('Image with mask', cv.bitwise_or(src_image, rgb_mask))
        cv.waitKey(0)

    #zastosowanie metody grabCut
    fg_model = np.zeros((1, 65), np.float64) #model pierwszego planu -> wymagane dla metody
    bg_model = np.zeros((1, 65), np.float64) #model tła -> wymagane dla metody
    iterations = 30 #liczba iteracji algorytmu grab cat (Ten parametr trzaba lepiej określić)
    mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model,
        iterations, mode=cv.GC_INIT_WITH_MASK)
    
    #wizualizacjia otrzymanej maski
    mask = np.where((mask==2)|(mask==0),0,255).astype('uint8')
    return mask

def crop_mask(oryginal_mask: UMat) -> UMat:
    middle_column_index = oryginal_mask.shape[1]//2

    #search from right to middle column
    right_columne_index = 0
    for col_index in range(middle_column_index):
        if 0 not in oryginal_mask[:, col_index]:
            right_columne_index = col_index
            break
    
    #search from left to middle column
    left_columne_index = oryginal_mask.shape[1]
    for col_index in reversed(range(middle_column_index, oryginal_mask.shape[1])):
        if 0 not in oryginal_mask[:, col_index]:
            left_columne_index = col_index
            break

    output_mask = np.zeros(oryginal_mask.shape, np.uint8)
    output_mask[:, right_columne_index:left_columne_index+1] = 255
    return output_mask

def sampling(image: UMat) -> list[UMat]:
    samples_list = [] # list for function to return
    sample_size = 50
    offset = 30
    x, y = 0, 0
    n= 0 
    while x+sample_size < image.shape[1]:
        while y+sample_size < image.shape[0]:
            sample = image[y:y+sample_size, x:x+sample_size]
            samples_list.append(sample)
            n+=1
            y+=offset
        y = 0
        x+=offset

    return samples_list