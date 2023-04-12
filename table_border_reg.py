import cv2
import numpy as np


def table_border_reg(img_path: str):
    image = cv2.imread(img_path, 1)
    # 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    rows, cols = binary.shape
    scale = 20
    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)

    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    # 膨胀
    kernel = np.ones((4, 4), np.uint8)
    img_dilate = cv2.dilate(merge, kernel, iterations = 1)
    return ~img_dilate


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_path = './imgs/test/65.png'
    result = table_border_reg(test_path)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite('result.jpg', result)
