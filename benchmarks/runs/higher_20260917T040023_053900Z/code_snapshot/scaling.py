"""Summarize the two baseline cases. Raw measurements remain in the run folder."""
import csv
import html
import json
import os
from pathlib import Path
import adapters
import numpy as np
import matplotlib.pyplot as plt
from present import hardware_summary

def report(folder):
    folder=Path(folder).resolve()
    out=folder/'report'; out.mkdir(exist_ok=True)
    meta=json.loads((folder/'metadata.json').read_text())
    with (folder/'raw_results.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if int(r['repeat'])>=0]
    summary=[]
    for case in ('moscap','sine'):
        for tool in ('devsim','scikit-fem'):
            previous=None
            for n in meta['meshes']:
                group=[r for r in rows if r['case']==case and r['tool']==tool and int(r['n'])==n]
                if len(group)!=meta['repeats']:
                    raise ValueError('Missing measurements')
                checked=next(r for r in group if r.get('relative_l2_integrated'))
                error=float(checked['relative_l2_integrated'])
                order=None
                if case=='sine' and previous:
                    order=float(np.log(previous[1]/error)/np.log(n/previous[0]))
                    if not 1.8<order<2.2:
                        raise ValueError('Sine error did not decrease at the expected rate')
                previous=(n,error)
                times=np.array([float(r['pipeline_s']) for r in group])
                summary.append(dict(case=case,tool=tool,n=n,free_dofs=int(group[0]['free_dofs']),
                    median_s=float(np.median(times)),q25_s=float(np.quantile(times,.25)),
                    q75_s=float(np.quantile(times,.75)),relative_error=error,convergence_order=order))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    colors={'devsim':'#126a9d','scikit-fem':'#da7732'}
    for case,style in [('moscap','o-'),('sine','s--')]:
        for tool,color in colors.items():
            group=[r for r in summary if r['case']==case and r['tool']==tool]
            sizes=[r['n'] for r in group]
            times=np.array([r['median_s'] for r in group])
            axes[0].errorbar(sizes,times,yerr=[times-[r['q25_s'] for r in group],
                [r['q75_s'] for r in group]-times],fmt=style,color=color,capsize=3,
                label=f'{"Regular" if case=="moscap" else "Sine"}: {tool}')
            if case=='sine':
                axes[1].plot(sizes,[r['relative_error'] for r in group],'o-',color=color,label=tool)
    for a in axes:
        a.set(xscale='log',yscale='log',xlabel='Mesh subdivisions per side')
        a.set_xticks(meta['meshes'],[str(n) for n in meta['meshes']])
        a.minorticks_off(); a.grid(alpha=.2); a.legend(fontsize=8)
    axes[0].set(title='More mesh points, more work',ylabel='Seconds per case; median and IQR')
    axes[1].set(title='Sine solution approaches the known answer',ylabel='Relative error')
    fig.tight_layout(); fig.savefig(out/'scaling.png',dpi=180); plt.close(fig)
    header=['Case','Tool']+[f'N={n}' for n in meta['meshes']]
    table=[]
    for case in ('moscap','sine'):
        for tool in colors:
            group=[r for r in summary if r['case']==case and r['tool']==tool]
            table.append(['Regular' if case=='moscap' else 'Sine',tool]+[f'{r["median_s"]:.2f} s' for r in group])
    intro='Two cases, unchanged equations, finer meshes. Both tools passed the analytical checks.'
    cases='Regular: the silicon–oxide capacitor, with its known straight-line voltage in each layer. Sine: u = sin(pi*x) sin(pi*y), a smooth field that varies in both directions.'
    timing=f'Median of {meta["repeats"]} measured runs per size, after an excluded warm-up. Includes mesh setup, solving and saving the same 512 × 512 output. No batch tests.'
    accuracy='The regular case stays near rounding accuracy. Sine error falls about fourfold when subdivisions double. Error is checked on the actual mesh, outside the timer.'
    hardware=hardware_summary(meta)
    codemap=Path(os.path.relpath(Path(__file__).parent/'CODE_MAP.md',out)).as_posix()
    md='# Two-case Poisson baseline\n\n'+intro+'\n\n'+cases+'\n\n'
    md+='| '+' | '.join(header)+' |\n|'+'---|'*len(header)+'\n'+''.join('| '+' | '.join(r)+' |\n' for r in table)
    md+=f'\n{timing}\n\n**Accuracy:** {accuracy}\n\n**Your computer:** {hardware}\n\n'
    md+=f'![Runtime and accuracy](scaling.png)\n\n[Code explained]({codemap}) · [Raw results](../raw_results.csv)\n'
    (out/'benchmark_summary.md').write_text(md,encoding='utf-8')
    esc=html.escape
    tablehtml='<tr>'+''.join(f'<th>{esc(v)}</th>' for v in header)+'</tr>'+''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in r)+'</tr>' for r in table)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-case Poisson baseline</title>
<style>body{{font:16px/1.5 "Segoe UI",Arial,sans-serif;color:#20323c;background:#f1f5f6;margin:0}}main{{max-width:1040px;margin:25px auto;background:white;padding:30px 40px;border-top:5px solid #167a80}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce5e8}}th{{background:#eef5f6}}img{{width:100%;height:auto}}small{{color:#526872}}a{{color:#126c77}}@media(max-width:650px){{main{{padding:15px}}table{{font-size:12px}}}}</style>
<main><h1>Two-case Poisson baseline</h1><p>{esc(intro)}</p><p>{esc(cases)}</p><table>{tablehtml}</table><p><small>{esc(timing)}</small></p><p><b>Accuracy:</b> {esc(accuracy)}</p><p><small><b>Your computer:</b> {esc(hardware)}</small></p><img src="scaling.png" alt="Measured runtime and sine accuracy at three mesh sizes"><p><a href="{esc(codemap)}">Code explained</a> · <a href="../raw_results.csv">Raw results</a></p></main></html>'''
    (out/'benchmark_summary.html').write_text(page,encoding='utf-8')

