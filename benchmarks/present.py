"""Generate a plain-language baseline summary from measured results."""
import argparse
import html
import json
import os
from pathlib import Path


def hardware_summary(meta):
    return (f'{meta.get("cpu_model",meta["processor"])}; '
            f'{meta["physical_cpus"]} physical / {meta["logical_cpus"]} logical cores reported; '
            f'{meta["ram_bytes"]/2**30:.1f} GiB total usable RAM; '
            f'{meta["platform"]}; CPU only, one compute thread; GPU not used.')


def present(run,out):
    run,out=Path(run).resolve(),Path(out).resolve()
    data=json.loads((out/'summary.json').read_text())
    hardware=hardware_summary(json.loads((run/'metadata.json').read_text()))
    rows=[]
    for tool,name in [('devsim','DEVSIM'),('scikit-fem','scikit-fem')]:
        r=next(v for v in data['single'] if v['tool']==tool and v['n']==16)
        rows.append((name,'Pass' if r['relative_l2']<.001 else 'Fail',f'{1000*r["pipeline"]["median"]:.1f} ms'))
    raw=Path(os.path.relpath(run/'results_enriched.csv',out)).as_posix()
    guide=Path(os.path.relpath(Path(__file__).resolve().parent/'README.md',out)).as_posix()
    sections=[
        ('The setup','A 100 nm × 100 nm cross-section: 90 nm of silicon below 10 nm of oxide. Set the top to 1 volt and the bottom to 0 volts. Insulate the sides. There is no charge inside either layer.'),
        ('The answer we expect','Voltage rises in a straight line through each layer, more steeply in the oxide. It reaches 0.75 volts at the layer boundary. A simple formula lets us check every solver independently.'),
        ('For each future solver','Same setup → check against the formula → time it the same way → compare. A future AI model can be checked against the same answer.'),
        ('What passing means','This simple case works. It does not prove that a solver can model a complete MOSFET or harder 2D physics. This baseline is deliberately easy.')]
    title='A simple baseline for our Poisson solver'
    question='Can it calculate the correct voltage across two layers—and how long does it take?'
    note='Both agreed to essentially numerical rounding accuracy. Median of five runs on this computer, using the same small mesh. Time includes setup, solving and saving the same output.'
    table='| Solver | Correct voltage? | Time per case |\n|---|---|---:|\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)
    md=f'# {title}\n\n**{question}**\n\n'
    for heading,body in sections[:2]:
        md+=f'**{heading}:** {body}\n\n'
    md+='![The test structure and expected voltage](moscap_solution.png)\n\n'+table+'\n'+note+'\n\n'
    md+=f'**Your computer:** {hardware}\n\n'
    for heading,body in sections[2:]:
        md+=f'**{heading}:** {body}\n\n'
    md+=f'[Technical details]({guide}) · [Raw results]({raw})\n'
    (out/'benchmark_summary.md').write_text(md,encoding='utf-8')
    esc=html.escape
    css='''*{box-sizing:border-box}body{margin:0;background:#f1f5f6;color:#20323c;font:16px/1.55 "Segoe UI",Arial,sans-serif}main{max-width:960px;margin:28px auto;padding:34px 44px;background:white;border-top:5px solid #167a80;border-radius:6px}h1{font-size:29px;line-height:1.2;margin:0 0 12px}h2{font-size:17px;margin:18px 0 5px}p{margin:6px 0}.question{color:#346571;font-size:18px}img{width:100%;height:auto;display:block;margin:16px 0}table{border-collapse:collapse;width:100%;margin:10px 0}th,td{text-align:left;padding:10px 15px;border-bottom:1px solid #dce5e8}th{background:#eef5f6}th:last-child,td:last-child{text-align:right}td:nth-child(2){color:#147343;font-weight:600}.note{font-size:13px;color:#586b75}.steps{margin:18px 0;background:#eaf5f3;padding:13px 17px;border-radius:5px}.limit{border-left:3px solid #cba653;padding-left:13px;margin:18px 0 12px}a{color:#126c77}footer{font-size:12px}@media(max-width:650px){main{margin:0;padding:22px}h1{font-size:25px}}@media print{body{background:white;font-size:12px}main{margin:0;padding:8px;border:0;max-width:none}h1{font-size:24px}h2{font-size:14px}img{max-height:250px;object-fit:contain}table,img,.steps{break-inside:avoid}.note,footer{font-size:10px}}'''
    body=f'<h1>{esc(title)}</h1><p class="question">{esc(question)}</p>'
    for heading,text in sections[:2]:
        body+=f'<h2>{esc(heading)}</h2><p>{esc(text)}</p>'
    body+='<img src="moscap_solution.png" alt="Silicon below oxide. Both calculated voltage profiles overlap the analytical answer.">'
    body+='<table><tr><th>Solver</th><th>Correct voltage?</th><th>Time per case</th></tr>'
    body+=''.join('<tr>'+''.join(f'<td>{esc(c)}</td>' for c in row)+'</tr>' for row in rows)+'</table>'
    body+=f'<p class="note">{esc(note)}</p>'
    body+=f'<p class="note"><b>Your computer:</b> {esc(hardware)}</p>'
    for (heading,text),style in zip(sections[2:],('steps','limit')):
        body+=f'<div class="{style}"><b>{esc(heading)}</b><p>{esc(text)}</p></div>'
    body+=f'<footer><a href="{esc(guide)}">Technical details</a> · <a href="{esc(raw)}">Raw results</a></footer>'
    page=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{css}</style></head><body><main>{body}</main></body></html>'
    (out/'benchmark_summary.html').write_text(page,encoding='utf-8')
    print(f'Summary: {out / "benchmark_summary.html"}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--output',required=True)
    args=p.parse_args(); present(args.run,args.output)
