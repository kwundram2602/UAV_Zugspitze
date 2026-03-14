import cairosvg, pathlib
base = pathlib.Path('Poster_UAV/Figures/analysis_hbbr_new')
for svg in base.rglob('*.svg'):
    pdf = svg.with_suffix('.pdf')
    cairosvg.svg2pdf(url=str(svg), write_to=str(pdf))
    print(f'Converted: {pdf}')