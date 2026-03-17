import cairosvg, pathlib

svg = pathlib.Path(r"D:\EAGLE\Zugspitze\UAV_Zugspitze\Poster_UAV\Figures\analysis_hbbr_new\chunk_metrics_table.svg")
pdf = svg.with_suffix('.pdf')
cairosvg.svg2pdf(url=str(svg), write_to=str(pdf))
print(f'Converted: {pdf}')