#### todolist
- 使用算法将表格的边缘补气
- 对于含有表格的照片表格无法监测（需要做二值化处理）

#### done
- 给表格图片增加白色边缘

#### progress
- 针对于不是照片类型的图片，利用二值化处理能够将图片中的其他颜色的噪声给过滤掉
```python
img = cv2.imread('./imgs/test/75.png')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 如果灰度图像中的像素值大于165，则将像素值设为255，否则将像素值设为0
# 而maxval参数表示最大像素值，也就是当像素值大于阈值时，设定的最大像素值。
# 使用的最大像素值是255，表示将像素值大于阈值的像素设为白色（即最大像素值。
ret, binary = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
plt.show()
```
- 可以对图片进行霍夫变换识别图像中的直线


alpha=1.0, 
benchmark=False, 
beta=1.0, 
cls_batch_num=6,
cls_image_shape='3, 48, 192',
cls_model_dir=None,
cls_thresh=0.9, 
cpu_threads=10, 
crop_res_save_dir='./output', 
det=True, 
det_algorithm='DB', 
det_box_type='quad', 
det_db_box_thresh=0.6,
det_db_score_mode='fast', 
det_db_thresh=0.3, 
det_db_unclip_ratio=1.5, 
det_east_cover_thresh=0.1, 
det_east_nms_thresh=0.2, 
det_east_score_thresh=0.8, 
det_limit_side_len=960, 
det_limit_type='max', 
det_model_dir='./models/ch_ppocr_server_v2.0_det_infer', 
det_pse_box_thresh=0.85, 
det_pse_min_area=16,
det_pse_scale=1, 
det_pse_thresh=0,
det_sast_nms_thresh=0.2,
det_sast_score_thresh=0.5, 
draw_img_save_dir='./inference_results', 
drop_score=0.5,
e2e_algorithm='PGNet', 
e2e_char_dict_path='./ppocr/utils/ic15_dict.txt',
e2e_limit_side_len=768,
e2e_limit_type='max',
e2e_model_dir=None,
e2e_pgnet_mode='fast',
e2e_pgnet_score_thresh=0.5, 
e2e_pgnet_valid_set='totaltext',
enable_mkldnn=False, 
fourier_degree=5, 
gpu_mem=500, 
help='==SUPPRESS==', 
image_dir=None,
image_orientation=False, 
ir_optim=True, 
kie_algorithm='LayoutXLM',
label_list=['0', '180'], 
lang='ch', 
layout=True, 
layout_dict_path='/home/zhuyuedlut/anaconda3/envs/paddle/lib/python3.8/site-packages/paddleocr/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt', 
layout_model_dir='/home/zhuyuedlut/.paddleocr/whl/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer',
layout_nms_threshold=0.5,
layout_score_threshold=0.5,
max_batch_size=10, 
max_text_length=25, 
merge_no_span_structure=True, 
min_subgraph_size=15, 
mode='structure', 
ocr=True, 
ocr_order_method=None,
ocr_version='PP-OCRv3',
output='./output', 
page_num=0, 
precision='fp32', 
process_id=0, 
re_model_dir=None, 
rec=True, 
rec_algorithm='SVTR_LCNet', 
rec_batch_num=6, 
rec_char_dict_path='/home/zhuyuedlut/anaconda3/envs/paddle/lib/python3.8/site-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt', 
rec_image_inverse=True, 
rec_image_shape='3, 48, 320', 
rec_model_dir='./models/ch_ppocr_server_v2.0_rec_infer',
recovery=False,
save_crop_res=False, 
save_log_path='./log_output/', 
scales=[8, 16, 32], 
ser_dict_path='../train_data/XFUND/class_list_xfun.txt', 
ser_model_dir=None,
show_log=True, 
sr_batch_num=1,
sr_image_shape='3, 32, 128', 
sr_model_dir=None, 
structure_version='PP-StructureV2', 
table=True, 
table_algorithm='TableAttn', 
table_char_dict_path='/home/zhuyuedlut/anaconda3/envs/paddle/lib/python3.8/site-packages/paddleocr/ppocr/utils/dict/table_structure_dict_ch.txt', 
table_max_len=488, 
table_model_dir='/home/zhuyuedlut/.paddleocr/whl/table/ch_ppstructure_mobile_v2.0_SLANet_infer', 
total_process_num=1, 
type='ocr', 
use_angle_cls=False, 
use_dilation=False, 
use_gpu=False, 
use_mp=False, 
use_npu=False, 
use_onnx=False, 
use_pdf2docx_api=False, 
use_pdserving=False, 
use_space_char=True, 
use_tensorrt=False, 
use_visual_backbone=True, 
use_xpu=False, 
vis_font_path='./doc/fonts/simfang.ttf', 
warmup=False