import glob
import cv2

images = glob.glob("images/*.jpg") # images 부분을 폴더 경로로 설정

for img_name in images:
    image = cv2.imread(img_name)
    image = cv2.resize(image, (600, 600))
    cv2.imwrite(img_name, images)