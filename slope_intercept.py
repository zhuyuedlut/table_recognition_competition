import cv2
import numpy as np

from xycut import bbox2points, vis_polygons_with_index


def calc_line(p0, p1, p2, p3):
    p_left_middle = [(p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2]
    p_right_middle = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

    # fit 一个多项式（deg=1）
    slope, intercept = np.polyfit(
        [p_left_middle[0], p_right_middle[0]],
        [p_left_middle[1], p_right_middle[1]],
        1
    )

    return slope, intercept


def calc_line_w_boxes(boxes):
    x_coords = []
    y_coords = []
    for p0, p1, p2, p3 in boxes:
        p_left_middle = [(p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2]
        p_right_middle = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        x_coords.append(p_left_middle[0])
        x_coords.append(p_right_middle[0])
        y_coords.append(p_left_middle[1])
        y_coords.append(p_right_middle[1])

    slope, intercept = np.polyfit(
        x_coords, y_coords, 1
    )
    return slope, intercept


def determine_is_in_same_line(
        p_left_middle, p_right_middle,
        slope, intercept,
        diff,
):
    y_left_ = p_left_middle[0] * slope + intercept
    y_right_ = p_right_middle[0] * slope + intercept

    diff_left = abs(y_left_ - p_left_middle[1])
    diff_right = abs(y_right_ - p_right_middle[1])

    if diff_left < diff:
        if diff_right < diff:
            return True

    return False



def combine_lines(list_ocr_results, indicators, length_thres=4):

    i = 0
    lines = []
    while i < len(list_ocr_results):
        if indicators[i] == 1:
            i += 1
            continue

        ocr_res = list_ocr_results[i]
        if len(ocr_res[1][0]) >= length_thres:
            tmp_line = [ocr_res]
            indicators[i] = 1

            # 看ocr结果中剩余的box是否和它属于同一行？
            for j, ocr_res_1 in enumerate(list_ocr_results):
                if indicators[j] == 1:
                    continue

                slope_, intercept_ = calc_line_w_boxes(
                    [w[0] for w in tmp_line]
                )

                p0, p1, p2, p3 = ocr_res_1[0]
                p_left_middle = [(p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2]
                p_right_middle = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

                diff = abs(p0[1] - p3[1]) * 0.75

                if determine_is_in_same_line(
                        p_left_middle, p_right_middle,
                        slope_, intercept_,
                        diff,
                ):
                    tmp_line.append(ocr_res_1)
                    indicators[j] = 1

            lines.append(
                tmp_line
            )

        i += 1

    return lines


def ocr_postprocess(paddle_ocr_results):

    if len(paddle_ocr_results) <= 1:
        return paddle_ocr_results

    # 1： 被处理过的box； 0：还没有被归纳为一行
    indicators = [0] * len(paddle_ocr_results)

    combined_lines = []
    for line in combine_lines(paddle_ocr_results, indicators, 3):
        combined_lines.append(line)

    for line in combine_lines(paddle_ocr_results, indicators, 1):
        combined_lines.append(line)

    for i in range(len(paddle_ocr_results)):
        if indicators[i] == 0:
            combined_lines.append(
                [paddle_ocr_results[i]]
            )

    # 对文字行进行排序
    for i, line in enumerate(combined_lines):
        line.sort(key=lambda x: x[0][0][0])
    combined_lines.sort(
        key=lambda x: x[0][0][0][1]
    )

    list_text_boxes = []
    for line in combined_lines:
        for w in line:
            list_text_boxes.append(w)

    return list_text_boxes


if __name__ == "__main__":

    # 调用paddle ocr
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    img_path = './imgs/test/14.png'
    image_idx = '1'
    ocr_result = ocr.ocr(img_path, cls=True)
    for line in ocr_result:
        print(line)

    boxes = []
    for line in ocr_result[0]:
        coords = line[0]

        x0 = coords[0][0]
        y0 = coords[0][1]
        x2 = coords[2][0]
        y2 = coords[2][1]

        boxes.append([int(x0), int(y0), int(x2), int(y2)])
    image = cv2.imread(img_path)

    # 采用slope-intercept重新排列后
    list_text_boxes = ocr_postprocess(ocr_result)
    boxes = []
    for line in list_text_boxes[0]:
        coords = line[0]
        x0 = coords[0][0]
        y0 = coords[0][1]
        x2 = coords[2][0]
        y2 = coords[2][1]
        boxes.append([int(x0), int(y0), int(x2), int(y2)])
    result = vis_polygons_with_index(image, [bbox2points(it) for it in boxes])
    cv2.imwrite(f"./output/paddleocr_slopeIntercept_{image_idx}.jpg", result)