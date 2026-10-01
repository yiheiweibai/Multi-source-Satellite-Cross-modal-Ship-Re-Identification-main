# 公开训练数据

完整带标签训练数据，不设置官方训练集和验证集划分。

## 目录结构

- `optical/`：光学图像
- `sar/`：SAR 图像
- `labels.csv`：唯一的标签文件

`labels.csv` 仅包含 `image_id`、`ship_id`、`modality` 和 `image_path` 四个字段。
其中，`image_path` 是相对于 `train/` 的路径。光学文件按 `opt_000001.tif` 格式命名，
SAR 文件按 `sar_000001.tif` 格式命名，训练身份按 `id_000001` 格式命名。

```csv
image_id,ship_id,modality,image_path
opt_000001,id_000001,optical,optical/opt_000001.tif
sar_000001,id_000001,sar,sar/sar_000001.tif
```
