"""The simple benchmark: two known problems, two tools, progressively finer meshes.

Run: python benchmarks/baseline.py             (128, 256, 512; five repeats)
     python benchmarks/baseline.py --quick     (16, 32, 64; one repeat)

The parent launches one worker at a time. A worker warms up, measures independent
solves, then exits. This keeps package state and memory separate between meshes.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from time import perf_counter

HERE=Path(__file__).resolve().parent
CASES=('moscap','sine')
TOOLS=('devsim','scikit-fem')
MESHES=(128,256,512)
REPEATS=5


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def worker(job):
    import adapters  # sets one compute thread before importing numerical libraries
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info
    from checks import relative_error,integrated_error
    if job['case']=='moscap':
        from moscap import ADAPTERS,field
        coefficients=[1.,0.]
    else:
        from adapters import ADAPTERS,field
        coefficients=[1.,0.,0.,0.]
    folder=Path(job['folder'])
    rows=[]
    for repeat in range(-1,job['repeats']):  # -1 is the excluded warm-up
        gc.collect()
        start=perf_counter()
        solver=ADAPTERS[job['tool']](job['n'])
        setup=dict(solver.times)
        solve=solver.solve(coefficients)
        sample_start=perf_counter()
        values=solver.sample(512)
        sample_s=perf_counter()-sample_start
        export_start=perf_counter()
        np.save(folder/f'field_{repeat}.npy',values,allow_pickle=False)
        export_s=perf_counter()-export_start
        pipeline_s=perf_counter()-start
        # Accuracy checks begin AFTER the complete pipeline timer stops.
        x,y=adapters.grid(512)
        error=relative_error(values,field(x,y,coefficients))
        stats=solver.stats()
        if not np.isfinite(values).all() or error >= (1e-10 if job['case']=='moscap' else .02):
            raise ValueError('Analytical accuracy check failed')
        if stats['boundary_max_abs_error']>=1e-10:
            raise ValueError('Boundary-value check failed')
        row=dict(case=job['case'],tool=job['tool'],n=job['n'],repeat=repeat,
                 pipeline_s=pipeline_s,sample_s=sample_s,export_s=export_s,
                 mesh_s=setup['mesh_s'],assembly_s=setup['assembly_s']+solve['assembly_s'],
                 solver_setup_s=setup['solver_setup_s'],solve_s=solve['solve_s'],
                 relative_l2_grid=error,**stats)
        if repeat==job['repeats']-1:
            row['relative_l2_integrated']=integrated_error(solver,field,coefficients,6)
            row['relative_l2_order8']=integrated_error(solver,field,coefficients,8)
            e6,e8=row['relative_l2_integrated'],row['relative_l2_order8']
            if job['case']=='sine' and abs(e8-e6)/e6>=.01:
                raise ValueError('Accuracy integration is not sufficiently resolved')
            if job['case']=='moscap' and max(e6,e8)>=1e-10:
                raise ValueError('Capacitor integrated accuracy check failed')
            if hasattr(solver,'residual'):
                row['algebraic_residual']=solver.residual()
                if row['algebraic_residual']>=1e-10:
                    raise ValueError('Algebraic residual check failed')
        rows.append(row)
        with (folder/'rows.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        save(folder/f'solver_{repeat}.json',solve)
        solver.close()
        del solver,values
    pools=threadpool_info()
    if any(pool['num_threads']!=1 for pool in pools):
        raise ValueError('A numerical library is using more than one thread')
    save(folder/'worker.json',dict(threadpools=pools,memory=psutil.Process().memory_info()._asdict(),
         memory_scope='solver worker lifetime; includes warm-up, interpolation and accuracy checks'))


def run(quick=False):
    import psutil
    import winreg
    run_id=('quick_' if quick else 'higher_')+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder=HERE/'runs'/run_id
    folder.mkdir(parents=True)
    meshes=(16,32,64) if quick else MESHES
    repeats=1 if quick else REPEATS
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    meta=dict(run_id=run_id,meshes=meshes,repeats=repeats,quick=quick,cpu_model=cpu,
        processor=platform.processor(),platform=platform.platform(),python=sys.version,
        physical_cpus=psutil.cpu_count(logical=False),logical_cpus=psutil.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,command=subprocess.list2cmdline([sys.executable,*sys.argv]),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        note='Sequential CPU workers; one compute thread each. Parent waits without polling. Background activity uncontrolled.')
    snapshot=folder/'code_snapshot'; snapshot.mkdir()
    meta['code_hashes']={}
    for name in ('baseline.py','adapters.py','moscap.py','problems.py','checks.py','scaling.py','present.py'):
        source=HERE/name
        shutil.copy2(source,snapshot/name)
        meta['code_hashes'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    save(folder/'metadata.json',meta)
    for case in CASES:
        for tool in TOOLS:
            for n in meshes:
                group=folder/f'{case}_{tool}_n{n}'; group.mkdir()
                job=dict(case=case,tool=tool,n=n,repeats=repeats,folder=str(group))
                save(group/'job.json',job)
                print(f'{case} / {tool} / N={n}',flush=True)
                with (group/'stdout.log').open('w',encoding='utf-8') as log:
                    completed=subprocess.run([sys.executable,str(HERE/'baseline.py'),'--worker',str(group/'job.json')],
                                             stdout=log,stderr=subprocess.STDOUT)
                save(group/'status.json',dict(exit_code=completed.returncode))
                if completed.returncode:
                    raise RuntimeError(f'Worker failed; logs retained at {group}')
    rows=[json.loads(line) for path in folder.glob('*/rows.jsonl') for line in path.read_text().splitlines()]
    with (folder/'raw_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader(); writer.writerows(rows)
    from scaling import report
    report(folder)
    print(f'\nComplete. Open: {folder / "report" / "benchmark_summary.html"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(json.loads(Path(args.worker).read_text()))
    else:
        run(args.quick)
