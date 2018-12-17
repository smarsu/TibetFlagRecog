# 基于 ResNet 实现的旗帜分类

## 目标
- 判断一张图片中是否存在目标旗帜, 为二分类任务

## 结果展示

![compare](https://github.com/smarsuuuuuuu/TibetFlagRecog/blob/master/compare.PNG)

- 左边为输入图片, 右边显示了模型分类所依据的区域
- 利用了 global pool 的性质, 能够高亮出待识别物体的大致位置, 在只有分类标签的情况下能实现一定程度的检测能力

## Start

### Download code
- `git clone https://github.com/smarsuuuuuuu/TibetFlagRecog.git`

### Download ckpt
- 链接: https://pan.baidu.com/s/1ciZwroN5clwozpbAtyea0A 提取码: 8sf1
- 下载后将文件解压在 ./TibetFlagRecog/ 目录下

### Show highlight
- 确保当前目录下有 test_visible.jpg
- `python visualization.py`
- 高亮结果会展示在窗口 (注意, 如果选择的图片里没有 TibetFlag, 高亮结果是无效的)
