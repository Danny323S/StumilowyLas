import numpy as np
import cv2 as cv

def main():
    while True:
        path_to_image = r'resources\drzewa\masks\dab\test_mask_dab(26).jpg'
        orygnial_mask = cv.imread(path_to_image)

        output_mask = orygnial_mask.copy()
        middle_column = output_mask.shape[1]//2

        #search from right to middle column
        right_columne_index = 0
        for col_index in range(middle_column):
            if 0 not in output_mask[:, col_index]:
                right_columne_index = col_index
                break
        cv.line(output_mask, (right_columne_index,0),
                (right_columne_index, output_mask.shape[0]), (0,0,255), 1)
        
        #search from left to middle column
        left_columne_index = 0
        for col_index in reversed(range(middle_column, output_mask.shape[1])):
            if 0 not in output_mask[:, col_index]:
                left_columne_index = col_index
                break
        cv.line(output_mask, (left_columne_index,0),
                (left_columne_index, output_mask.shape[0]), (0,0,255), 1)

        cv.imshow('Oryginal mask', orygnial_mask)
        cv.imshow('Outpu mask', output_mask)
        key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            exit()

if __name__ == "__main__":
    main()