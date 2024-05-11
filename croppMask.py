import numpy as np
import cv2 as cv

def main():
    # while True:
        save_output = False
        for i in range(1,90):
            # path_to_image = r'resources\drzewa\masks\dab\test_mask_dab(26).jpg'
            path_to_image = fr'output_fils\masks\dab\mask-dab_({i}).jpg'
            # path_to_image = fr'output_fils\masks\sosna\mask-sosna_({i}).jpg'

            orygnial_mask = cv.imread(path_to_image)

            output_mask = orygnial_mask.copy()


            middle_column = output_mask.shape[1]//2
            final_mask = np.zeros_like(orygnial_mask[:,:,0])
            #search from right to middle column
            right_columne_index = 0
            for col_index in range(middle_column):
                if 0 not in output_mask[:, col_index]:
                    right_columne_index = col_index
                    break
            cv.line(output_mask, (right_columne_index,0),
                    (right_columne_index, output_mask.shape[0]), (0,0,255), 1)
            cv.line(final_mask, (right_columne_index,0),
                    (right_columne_index, output_mask.shape[0]), (255,255,255), 1)
            
            #search from left to middle column
            left_columne_index = 0
            for col_index in reversed(range(middle_column, output_mask.shape[1])):
                if 0 not in output_mask[:, col_index]:
                    left_columne_index = col_index
                    break
            cv.line(output_mask, (left_columne_index,0),
                    (left_columne_index, output_mask.shape[0]), (0,0,255), 1)
            cv.line(final_mask, (left_columne_index,0),
                    (left_columne_index, output_mask.shape[0]), (255,255,255), 1)
            
            
            final_mask = cv.rectangle(final_mask, (left_columne_index,0), 
                                      (right_columne_index, output_mask.shape[0]),(255,255,255),thickness=cv.FILLED)
            # cv.imshow('Oryginal mask', orygnial_mask)
            # cv.imshow('Outpu mask', output_mask)
            # cv.imshow('Final mask', final_mask)
            
            if save_output == True:
                output_file_name = 'new' + path_to_image.split('\\')[-1]
                output_diectory = ('maski\\' + path_to_image.split('\\')[-2]) + '\\'
                cv.imwrite(output_diectory+output_file_name, final_mask)
                # print(output_diectory)
                # print(output_file_name)

            key = cv.waitKey(0)
            if key == ord('q') or key == ord('Q') or key == 27:
                exit()
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()