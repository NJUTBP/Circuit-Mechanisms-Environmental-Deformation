# Figs_CTP

- **Main_figs**: Main manuscript figures.
- **Supp_figs**: Supplementary figures.
- **Plot_data**: Final plotting data and graphical assets, organized by figure number.
- **Plot_code**: Individual plotting scripts, shared style functions, and Python dependencies. Data paths point to `Plot_data` within this folder.

Run `python Plot_code/plot_all.py` to regenerate all figures. Main Figure 1 uses the supplied PDF artwork and requires Poppler for raster export. The other figures are drawn from numerical data. The font is Times New Roman; PNG files are exported at 600 dpi.
