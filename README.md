# luzt-mmdet
my mmdet

### mmdetection的修改
#### 数据增强：
- [x] mixup
- [x] mosaic
- [x] 类似Stitcher中的mosaic，代码中标记为masaic
- [ ] bboxjitter
- [x] gridmask(非训练版本)
- [x] Minus（减去模板的均值或序列图片均值）
#### 模型修改：
- [x] 新增bifpn实现
- [x] global roi 
- [ ] atss_Rcnn(代码可能有问题)
- [x] repulsion loss
- [ ] diou loss & ciou loss （需要进一步修改，指标偏低）
- [x] senet 
### data_make 
- [x] json2voc and voc2coco
- [x] duck injucktion
- [ ] make_gt_json
- [x] 反色数据
- [x] 训练验证集分割

### data_analysis
- [x] 可视化json
- [x] 可视化xml
- [x] 可视化每个类别的位置分布
- [x] 计算长宽比，大中小目标数量分布，各个类别数量分布
- [x] 把多个结果图片拼接起来对比
- [x] 多个结果文件的bbox打到一张图上和gt对比
- [ ] 结果fp和fn的可视化
### 模型融合
- [x] 加权平均
- [x] 软加权平均


hilens
链接：https://pan.baidu.com/s/1snwYMBTatRs1lX-af1xzKg 
提取码：s21b
