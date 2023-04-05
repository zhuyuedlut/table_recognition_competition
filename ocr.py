import cv2

from paddleocr import PaddleOCR
from utils import vis_polygons_with_index, bbox2points

if __name__ == '__main__':
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch'
    )

    img_path = './imgs/test/14.png'
    image_idx = '1'

    ocr_result = ocr.ocr(img_path, cls=True)

    boxes = []
    result = ocr_result[0]
    for line in result:
        coords = line[0]

        x0 = coords[0][0]
        y0 = coords[0][1]
        x2 = coords[2][0]
        y2 = coords[2][1]

        boxes.append([int(x0), int(y0), int(x2), int(y2)])

    image = cv2.imread(img_path)
    result = vis_polygons_with_index(image, [bbox2points(it) for it in boxes])
    cv2.imwrite(f'./output/paddleocr_original_{image_idx}.jpg', result)

