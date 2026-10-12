# FNO 使用说明

本目录包含论文 FNO 部分的三场自回归正演、训练与预测、GA 参数反演、可选 L-BFGS
反演，以及 COMSOL/CADET 接口。论文报告的 FNO 反演主要使用 GA 后接 COMSOL 复算筛选。

本次仅公开代码、配置和合成示例生成器；不附带原始实验数据、FEM 训练集或预训练权重。
示例使用人工高斯曲线，只用于验证程序流程，不能复现论文中的精度或证明参数可辨识。

## 快速运行

建议使用 Python 3.11，在本目录创建独立环境并运行：

```bash
python -m venv .venv
# Windows PowerShell 激活：.venv\Scripts\Activate.ps1
# Linux/macOS 激活：source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python examples/run_smoke.py
```

程序会生成小型数据，完成训练、保存与加载、预测以及两种反演。输出在 `logs/` 和
`results/`。默认正式模型为 101 个轴向节点、6 个径向节点、360 个时间点，四层 FNO，
隐藏宽度 128。CPU 示例会缩小网络和网格。

训练正式数据：

```bash
python -m src.train data.h5_file_path=/path/to/train.h5
```

训练后迁移模型时，必须一起保留 `checkpoints/`、`.hydra/config.yaml` 和
`.hydra/para_info.yaml`。参数输入使用真实物理值，不能直接传归一化坐标或对数值。

GA 反演的目标文件包含 `outlet (时间点,实验数)`、`end_con (实验数,)` 和
`end_time (实验数,)`。同一目标文件中的曲线应来自同一物质并共享九个物理参数。
实验强度需要先完成有记录的浓度标定和进样归一化。

```bash
python -m src.inverse --method ga --checkpoint /path/to/run/checkpoints/best.ckpt --target /path/to/target.npz
```

COMSOL 接口依赖合法安装的 COMSOL 和对应模型。默认等度验证不改变 `kd`；论文中的
二倍 `kd` 协议需要明确传入 `--isocratic-kd-scale 2`，并会记录到结果中。该调整不能
被描述成所有物理参数完全不变的迁移验证。

详细参数和命令见 [英文说明](../README.md)、[数据格式](data_format.md)、
[方法说明](method.md)、[代码整理差异](source_mapping.md)和[验证记录](validation.md)。
