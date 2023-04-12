# -*- coding: utf-8 -*-
"""
# @file name  : main.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/22 13:20
# @brief      :
"""
import os
import cv2
import numpy as np
import copy

from paddleocr import PPStructure, save_structure_res
from paddleocr.ppstructure.table.matcher import TableMatch

from table_border_reg import table_border_reg
from table_ocr import ocr_table_img
from slope_intercept import ocr_postprocess

if __name__ == "__main__":
    table_engine = PPStructure(
        layout=False,
        image_orientation=True,
        show_log=True,
        det_db_score_mode='slow',
        label_list=['0', '90', '180', '270'],
    )

    save_folder = './output'
    img_path = './imgs/test/33.png'

    # 提取表格边框
    table_border = table_border_reg(img_path)
    # plt.imshow(cv2.cvtColor(table_border, cv2.COLOR_BGR2RGB))
    # plt.show()
    temp_table_border_result = table_engine(table_border, return_ocr_result_in_table=False)
    save_structure_res(temp_table_border_result, './output/border', os.path.basename(img_path).split('.')[0])
    structure_res, elapse = table_engine.table_system.table_structurer(copy.deepcopy(temp_table_border_result[0]['img']))

    # 提取表格中的文字
    img = cv2.imread(img_path)
    table_text = ocr_table_img(img_path)
    post_table_text = [ocr_postprocess(table_text[0])]
    dt_boxes = []
    rec_res = []
    for line in post_table_text[0]:
        coords = line[0]
        dt_boxes.append(np.array(coords[0] + coords[2]))
        rec_res.append(np.array(line[1]))

    # 将表格边框和文字进行匹配
    match = TableMatch(filter_ocr_result=True)
    pred_html = match(structure_res, dt_boxes, rec_res)
    print(pred_html)

    final_result = copy.deepcopy(temp_table_border_result)
    final_result[0]['res']['html'] = pred_html
    save_structure_res(final_result, save_folder, os.path.basename(img_path).split('.')[0])


