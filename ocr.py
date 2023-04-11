import cv2

from paddleocr import PaddleOCR, PPStructure
from utils import vis_polygons_with_index, bbox2points

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch'
)


def ocr_img(img_path: str):
    result = ocr.ocr(img_path, cls=True)

    return result


if __name__ == '__main__':
    test_img_path = './imgs/test/14.png'
    ocr_result = ocr.ocr(test_img_path, cls=True)
    boxes = []
    for line in ocr_result[0]:
        coords = line[0]

        x0 = coords[0][0]
        y0 = coords[0][1]
        x2 = coords[2][0]
        y2 = coords[2][1]

        boxes.append([int(x0), int(y0), int(x2), int(y2)])

    image = cv2.imread(test_img_path)
    image_with_box = vis_polygons_with_index(image, [bbox2points(it) for it in boxes])
    cv2.imwrite(f'./output/paddleocr_original_test.jpg', image_with_box)
