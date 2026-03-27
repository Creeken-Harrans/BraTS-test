# BraTS 第一个病例可视化结果

本目录由 `visualize_first_brats_case.py` 自动生成。

- 数据根目录：`/home/creeken/Desktop/deep-learning/archive`
- 当前自动选中的病例目录：`/home/creeken/Desktop/deep-learning/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001`
- 输出目录：`/home/creeken/Desktop/deep-learning/BraTS-test/visualize/result`

## 为什么 BraTS 数据不是 jpg/png，而是 3D NIfTI

BraTS 使用的是医学影像常见的 **NIfTI** 格式，因为它保存的是完整的三维 MRI 体数据，而不是单张二维图片。

- 一份 NIfTI 文件通常对应一个 3D volume，例如 `(240, 240, 155)` 这样的体素网格。
- 每个体素不仅有强度值，还带有空间信息，例如 **voxel spacing** 和 **affine**。
- 这样我们才能在 **axial / coronal / sagittal** 三个方向上切片观察，而不是只看一张静态图。
- 分割标签 `seg` 也是一个与 MRI 完全对齐的 3D 体数据，所以可以做逐体素 overlay、3D bounding box 和标签统计。

## 模态说明

- `t1`：T1 加权 MRI，适合看基础解剖结构。
- `t1ce`：T1 对比增强 MRI，也常写作 `t1gd`，增强区通常更明显。
- `t2`：T2 加权 MRI，液体相关信号更亮。
- `flair`：抑制脑脊液后的 T2 风格图像，常更利于观察水肿区域。
- `seg`：分割标签体数据，不是普通图片，而是每个体素一个类别值。

## 输出文件说明

- `case_summary.txt`：病例路径、文件路径、shape、dtype、voxel spacing、affine、seg 标签统计的可读文本版摘要。
- `case_summary.json`：与文本摘要对应的结构化 JSON，便于程序继续处理。
- `intensity_stats.csv`：四个模态的强度统计，包括全体素和非零区域统计。
- `modalities_mid_slices.png`：四个模态在 axial / coronal / sagittal 三个方向的中间切片总览。
- `t1_montage.png`：T1 的 axial 多切片 montage，用来观察 3D 体数据沿 z 方向的变化。
- `t1ce_montage.png`：T1CE 的 axial 多切片 montage。
- `t2_montage.png`：T2 的 axial 多切片 montage。
- `flair_montage.png`：FLAIR 的 axial 多切片 montage。
- `seg_labels_summary.txt`：seg 标签值解释及体素数量统计。
- `seg_montage.png`：seg 标签的多切片可视化，使用离散颜色区分标签。
- `flair_overlay_best_slices.png`：在 FLAIR 上叠加 seg，自动选择最能体现肿瘤的三视图切片。
- `t1ce_overlay_best_slices.png`：在 T1CE 上叠加 seg，观察增强区域与标签的对应关系。
- `tumor_bbox_views.png`：根据 seg 计算 3D tumor bounding box，并展示整图视角与局部放大视角。
- `intensity_histograms.png`：四个模态的强度分布直方图，分别展示全体素和非零体素。
- `seg_label_distribution.png`：seg 标签在全 volume 中的占比，以及在非背景 tumor voxel 中的占比。

## 学习建议

- 先看 `case_summary.txt`，建立对 shape、spacing、affine 的基本认识。
- 再看 `modalities_mid_slices.png` 和各个 montage，理解同一病例在不同模态、不同切片中的表现差异。
- 最后看 overlay 和 bounding box 图，体会分割标签是如何与 3D MRI 体数据对齐的。
