# Circuit mechanisms shaping place- and grid-cell representations under environmental deformation

本项目提供论文配套的模型代码、分析代码和绘图数据，投稿期刊为 *Communications in Theoretical Physics*。

## 绘图

使用 Python 3.11，在项目根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-plot.txt
python Figs_CTP/Plot_code/plot_all.py
```

macOS/Linux 使用 `source .venv/bin/activate` 激活环境。
主图 1 需要安装 Poppler，将 `pdftoppm` 加入 PATH，或通过 `PDFTOPPM` 指定其路径。
绘图字体使用 Times New Roman，输出保存在 `Figs_CTP/Reproduced/`。

只绘制指定图：

```powershell
python Figs_CTP/Plot_code/plot_all.py --figures main_figure_2
```

## 目录

| 内容 | 路径 |
| --- | --- |
| 模型与参数配置 | `codes/run/` |
| 分析与统计代码 | `codes/analysis/` |
| 绘图脚本 | `Figs_CTP/Plot_code/` |
| 按图号整理的数据 | `Figs_CTP/Plot_data/` |
| 6 张主图、16 张补充图 | `Figs_CTP/Main_figs/`、`Figs_CTP/Supp_figs/` |
| 图与文件对应表 | [docs/DATA_INDEX.md](docs/DATA_INDEX.md) |
| 数据字段与数组维度 | [docs/DATA_FILES.csv](docs/DATA_FILES.csv) |

## 模型与分析

另建 Python 环境并安装模型与分析依赖：

```powershell
python -m pip install -r requirements-analysis.txt
python codes/run/run_pipeline.py --help
```

具体用法见[模型说明](codes/run/README.md)和[分析说明](codes/analysis/README.md)。

## 许可证

[MIT](LICENSE)。
