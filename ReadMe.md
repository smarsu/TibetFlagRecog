# 基于 ResNet 实现的旗帜分类

## 目标
- 判断一张图片中是否存在目标旗帜, 为二分类任务

## 结果展示

![compare](https://github.com/smarsuuuuuuu/TibetFlagRecog/blob/master/compare.PNG)

- 左边为输入图片, 右边显示了模型分类所依据的区域
- 利用了 global pool 的性质, 能够高亮出待识别物体的大致位置, 在只有分类标签的情况下能实现一定程度的检测能力
