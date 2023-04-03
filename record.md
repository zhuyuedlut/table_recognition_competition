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