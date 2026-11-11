from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

from run_cell import ROOT,PYTHON,environment
from freetoken.moe.tablemoe_artifacts import read_json,write_json

DATASETS=['MMLongBench_DOC','BabyVision','Vision2Web_Level1','MMMU_DEV_VAL']
METHODS=['fused','tablemoe','modes','schedule','table']


def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(')')[1].split()[0]
        return state!='Z'
    except FileNotFoundError:
        return False


def resources():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    values={}
    for line in raw.splitlines():
        i,uuid,free,util=[p.strip() for p in line.split(',')]
        values[int(i)]=dict(uuid=uuid,free_mib=int(free),utilization=int(util),pids=[])
    for line in apps.splitlines():
        pid,uuid=[p.strip() for p in line.split(',')]
        for v in values.values():
            if v['uuid']==uuid:v['pids'].append(int(pid))
    return values


def jobs():
    result=[]
    def add(id,cmd,marker,deps=(),priority=10):
        result.append(dict(id=id,command=cmd,marker=str(marker),dependencies=list(deps),priority=priority))
    for dataset in DATASETS:
        prepared=str(ROOT/'prepare'/dataset/'complete.json')
        if dataset!='MMMU_DEV_VAL':
            build_deps=[prepared]
            if dataset in ('MMLongBench_DOC','Vision2Web_Level1'):
                build_deps.append(str(ROOT/'validation/compact_storage_ready.json'))
            add('build_'+dataset,[PYTHON,'-B',str(ROOT/'build_table.py'),dataset,'{gpu}'],
                ROOT/'build'/dataset/'complete.json',build_deps,0)
        asset=str(ROOT/'build'/dataset/'complete.json') if dataset!='MMMU_DEV_VAL' else prepared
        add('calibrate_'+dataset,[PYTHON,'-B',str(ROOT/'calibrate_modes.py'),dataset],
            ROOT/'calibration'/dataset/'prefill_decode/complete.json',[asset] if dataset=='Vision2Web_Level1' else [prepared],7)
        for method in METHODS:
            deps=[prepared]
            if method in ('tablemoe','table'):deps.append(asset)
            if method=='modes':deps.append(str(ROOT/'calibration'/dataset/'prefill_decode/complete.json'))
            if dataset=='Vision2Web_Level1':
                cmd=[PYTHON,'-B',str(ROOT/'vision_pipeline.py'),dataset,method,'{gpu}','inference','--deadline','1794450189']
                marker=ROOT/'vision'/method/'generation_complete.json'
                deps.append(str(ROOT/'validation/vision_pipeline_ready.json'))
            else:
                cmd=[PYTHON,'-B',str(ROOT/'run_cell.py'),dataset,method,'--gpu','{gpu}']
                marker=ROOT/'cells'/dataset/method/'generation_complete.json'
            add('generate_'+dataset+'_'+method,cmd,marker,deps,dict(fused=5,tablemoe=6,modes=5,schedule=8,table=9)[method])
    if (ROOT/'additional_jobs.json').exists():
        result.extend(read_json(ROOT/'additional_jobs.json'))
    policy=ROOT/'scheduling_policy.json'
    if policy.exists():
        values=read_json(policy)
        priorities=values.get('priorities',{})
        commands=values.get('commands',{})
        dependencies=values.get('additional_dependencies',{})
        for job in result:
            job['priority']=priorities.get(job['id'],job['priority'])
            job['command']=commands.get(job['id'],job['command'])
            job['dependencies'] += dependencies.get(job['id'],[])
    return sorted(result,key=lambda j:j['priority'])


def complete(path):
    p=Path(path)
    return p.exists() and read_json(p).get('complete',False)


def stop_owned(pid):
    descendants=set([pid])
    for _ in range(8):
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                try:
                    parent=int((p/'stat').read_text().split(')')[1].split()[1])
                    if parent in descendants:descendants.add(int(p.name))
                except (OSError,ValueError,IndexError):pass
    descendants.discard(os.getpid())
    stamps={}
    for child in descendants:
        try:stamps[child]=Path(f'/proc/{child}/stat').read_text().split(')')[1].split()[19]
        except (OSError,IndexError):pass
    def remaining():
        result=[]
        for child,stamp in stamps.items():
            try:
                current=Path(f'/proc/{child}/stat').read_text().split(')')[1].split()
                if current[19]==stamp and current[0]!='Z':result.append(child)
            except (OSError,IndexError):pass
        return result
    signals=[]
    for sig,grace in [(signal.SIGINT,45),(signal.SIGTERM,15),(signal.SIGKILL,5)]:
        live=remaining()
        if not live:break
        for child in sorted(live,reverse=True):
            try:os.kill(child,sig)
            except ProcessLookupError:pass
        signals.append(dict(signal=int(sig),pids=live))
        until=time.time()+grace
        while remaining() and time.time()<until:time.sleep(1)
    live=remaining()
    return dict(requested=sorted(stamps),confirmed_stopped=sorted(set(stamps)-set(live)),still_alive=live,signals=signals)


def main():
    from report_results import main as report
    lock=(ROOT/'manager.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy=read_json(ROOT/'execution_policy.json')
    deadline=policy['deadline_unix']
    state=read_json(ROOT/'status.json') if (ROOT/'status.json').exists() else dict(jobs={},workers={})
    state.update(status='running',started_unix=state.get('started_unix',time.time()),deadline_unix=deadline)
    active={int(gpu):job for gpu,job in state['workers'].items() if alive(job['pid'])}
    for gpu,file,id in [(0,'mmmu_fused_job.json','generate_MMMU_DEV_VAL_fused'),(1,'babyvision_build_job.json','build_BabyVision')]:
        existing=read_json(ROOT/file)
        if alive(existing['pid']):active[gpu]=dict(id=id,pid=existing['pid'],adopted=True)
    processes={}
    while time.time()<deadline:
        queue=jobs()
        for gpu,job in list(active.items()):
            process=processes.get(job['id'])
            done=process.poll() is not None if process else not alive(job['pid'])
            if done:
                spec=next(j for j in queue if j['id']==job['id'])
                success=complete(spec['marker'])
                state['jobs'][job['id']]={**job,'status':'complete' if success else 'failed',
                    'exit_code':process.returncode if process else None,'finished_unix':time.time()}
                del active[gpu]
        resource=resources()
        for gpu in [0,1]:
            if gpu in active or resource[gpu]['pids'] or resource[gpu]['free_mib']<76*1024:continue
            active_ids={v['id'] for v in active.values()}
            ready=[j for j in queue if j['id'] not in active_ids and j['id'] not in state['jobs']
                and not complete(j['marker']) and all(complete(d) for d in j['dependencies'])]
            if not ready:continue
            if ready[0]['id'].startswith('build_') and shutil.disk_usage('/data/fanya').free<60*(1<<30):
                state['storage_gate']=dict(free_bytes=shutil.disk_usage('/data/fanya').free,required_bytes=60*(1<<30))
                ready=[j for j in ready if not j['id'].startswith('build_')]
                if not ready:continue
            job=ready[0]
            cmd=[p.replace('{gpu}',str(gpu)).replace('{smoke_deadline}',str(min(deadline,time.time()+1800))) for p in job['command']]
            env=environment(gpu)
            log=ROOT/'logs'/(job['id']+'.log')
            attempt=1
            while log.exists():
                log=ROOT/'logs'/(job['id']+f'_attempt_{attempt:03d}.log')
                attempt+=1
            with log.open('x') as output:
                proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=dict(id=job['id'],pid=proc.pid,started_unix=time.time(),command=cmd,log=str(log))
            processes[job['id']]=proc
        state.update(workers={str(k):v for k,v in active.items()},resources=resource,updated_unix=time.time(),
            jobs_complete=[j['id'] for j in queue if complete(j['marker'])],
            pending=[j for j in queue if not complete(j['marker']) and j['id'] not in state['jobs']])
        write_json(ROOT/'status.json',state)
        report()
        with (ROOT/'resource_observations.jsonl').open('a') as f:
            f.write(json.dumps(dict(timestamp=time.time(),gpus=resource,workers=state['workers']))+'\n')
        if not active and all(complete(j['marker']) or j['id'] in state['jobs'] for j in queue):
            break
        time.sleep(10)
    stopped=[]
    for job in active.values():stopped.append(stop_owned(job['pid']))
    state.update(status='deadline_reached' if time.time()>=deadline else 'queue_finished',
        shutdown_receipts=stopped,finished_unix=time.time(),complete=all(complete(j['marker']) for j in jobs()))
    write_json(ROOT/'status.json',state)
    write_json(ROOT/'execution_end.json',state)
    report()


if __name__=='__main__':
    main()
