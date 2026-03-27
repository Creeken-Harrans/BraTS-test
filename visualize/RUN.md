# 运行说明

## 运行命令

```bash
python3 /home/creeken/Desktop/deep-learning/BraTS/visualize/visualize_first_brats_case.py
```

脚本默认使用固定路径：

- `DATA_ROOT = "/home/creeken/Desktop/deep-learning/archive"`
- `OUTPUT_DIR = "/home/creeken/Desktop/deep-learning/BraTS/visualize/result"`

## 依赖库名称

- `nibabel`
- `numpy`
- `matplotlib`
- `pandas`

说明：

- 脚本优先使用 `nibabel` 读取 NIfTI，优先使用 `pandas` 写统计表
- 如果当前环境缺少 `nibabel` 或 `pandas`，脚本会自动回退到内置的轻量 NIfTI/CSV 逻辑
- 但 `numpy` 和 `matplotlib` 仍然是运行所必需的

## 输出目录说明

- 输出目录固定为：`/home/creeken/Desktop/deep-learning/BraTS/visualize/result`
- 脚本运行时会自动创建该目录
- 结果会写入文本摘要、JSON 摘要、CSV 统计表，以及多个 PNG 可视化图
- `README.md` 也会自动生成在该目录中，说明每个输出文件的作用

## 常见报错排查提示

- `Missing required Python packages: ...`
  说明当前 Python 环境缺少关键依赖库。当前脚本对 `nibabel` 和 `pandas` 有内置回退，但 `numpy` 和 `matplotlib` 仍必须存在。

- `Data root does not exist`
  检查 `/home/creeken/Desktop/deep-learning/archive` 是否存在，以及路径是否有误。

- `No BraTS-style patient directory was found`
  说明脚本在 `archive` 下没有扫描到符合 BraTS 风格的病例目录。请确认数据是否已经解压，并且目录中包含 `.nii` 或 `.nii.gz` 文件。

- `The first patient directory in sorted order is incomplete`
  说明按路径排序后的第一个病例目录缺少 `t1 / t1ce(or t1gd) / t2 / flair / seg` 中的某些文件，或者文件命名不规范。

- `Volumes are not aligned in the same 3D space`
  说明不同模态或分割标签的 shape、spacing、affine 不一致，overlay 结果不可靠，需要先检查数据是否配准一致。

- `Segmentation contains only background label 0`
  说明当前 `seg` 中没有肿瘤标签，因此无法生成肿瘤 overlay、bounding box 等以肿瘤区域为中心的图。
