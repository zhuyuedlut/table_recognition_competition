#### 默认使用的模型列表
- rec : ch_PP-OCRv3_rec_infer
- dec : ch_PP-OCRv3_det_infer
- table : ch_ppstructure_mobile_v2.0_SLANet_infer
- layout: picodet_lcnet_x1_0_fgd_layout_cdla_infer

#### 使用改进的模型列表
- rec: 
- dec: ch_ppocr_server_v2.0_det_infer

#### todolist
- 使用算法将表格的边缘补气
- 对于含有表格的照片表格无法监测（需要做二值化处理）

#### done
- 给表格图片增加白色边缘