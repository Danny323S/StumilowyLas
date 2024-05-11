import cv2 as cv
import numpy as np


def main():
    for i in range(1,90):
        # path_to_image = r'resources\drzewa\masks\dab\test_mask_dab(26).jpg'
        path_to_image = fr'output_fils\masks\dab\mask-dab_({i}).jpg'
        orygnial_mask = cv.imread(path_to_image)

        output_mask = orygnial_mask.copy()

        output_mask_gray = cv.cvtColor(output_mask, cv.COLOR_BGR2GRAY)
        kernel = np.ones((7,7), np.uint8)
        output_mask_gray = cv.morphologyEx(output_mask_gray, cv.MORPH_CLOSE, kernel)
        contours,_ = cv.findContours(output_mask_gray, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        cnt = contours[0]
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        # print(box)
        # print([box])
        
        #draw minimum area rectangle (rotated rectangle)
        output_mask = cv.drawContours(output_mask,[box],0,(0,255,255),2)

        rect_width = rect[1][0]
        rect_height = rect[1][1]
        angle = rect[2] if rect_width < rect_height else rect[2]-90
        height, width = output_mask.shape[:2]
        center = (width / 2, height / 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # Wykonujemy transformację obrazu
        rotated_mask = cv.warpAffine(output_mask, M, (width, height))
        # rotated_mask = cv.morphologyEx(rotated_mask, cv.MORPH_CLOSE, kernel)

        middle_column = rotated_mask.shape[1]//2

        #search from right to middle column
        right_columne_index = 0
        for col_index in range(middle_column):
            if 0 not in rotated_mask[:, col_index]:
                right_columne_index = col_index
                break
        cv.line(rotated_mask, (right_columne_index,0),
                (right_columne_index, rotated_mask.shape[0]), (0,0,255), 1)
        
        #search from left to middle column
        left_columne_index = 0
        for col_index in reversed(range(middle_column, rotated_mask.shape[1])):
            if 0 not in rotated_mask[:, col_index]:
                left_columne_index = col_index
                break
        cv.line(rotated_mask, (left_columne_index,0),
                (left_columne_index, rotated_mask.shape[0]), (0,0,255), 1)

        # alpha_l = 0
        # # szukamy wartość przesunięcia alpha
        # while True:
        #     x1 = int(box[0][0]+alpha_l)
        #     y1 = box[0][1]

        #     x2 = int(box[1][0]+alpha_l)
        #     y2 = box[1][1]

        #     line_region = output_mask_gray[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        #     # Sprawdź, czy w obszarze linii są jakieś piksele o wartościach 0
        #     if np.any(line_region == 0):
        #         alpha_l+=1
        #         print(alpha_l)
        #     else:
        #         print("Linia  przechodzi tylko przez obszar o wartości 1 w masce.")
        #         cv.line(output_mask, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #         break

        
        #wyswietlanie masek
        cv.imshow('Oryginal mask', orygnial_mask)
        cv.imshow('Output mask', output_mask)
        cv.imshow('rotated mask', rotated_mask)

        key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            exit()
        cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()