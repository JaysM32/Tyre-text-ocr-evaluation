import cv2
import numpy as np



for i in range(4):
    n = i+20+7
    

    b = "bright"
    d = "dark"




    image = cv2.imread(f"additions/text_recog/img-{n}-{b}-raw.png")
    print(n)
 
    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    # tileGridSize -> Number of tiles in the row and column
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
    final_img = clahe.apply(image_bw)

    # Adjust the brightness of the output
    final_img = cv2.add(final_img, 40)
 
    cv2.imwrite(f"additions/text_recog/img-{n+10}-preprocesed.png", final_img, params=None)  # save unwrap image to file.