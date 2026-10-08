// Opt-in scaling check: real streamed preparation and CPU fits, no agent subscription.
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { runProcess } from './process.mjs';
import { inspectDataset, evaluateCandidate } from './evaluation.mjs';
import { cleanupContainers } from './runtime.mjs';
const root=await mkdtemp(join(tmpdir(),'helix-scaling-'));
const python=process.env.HELIX_PYTHON || 'python3';
const task={agent:'codex',dataset:join(root,'million.csv'),target:'label',objective:'Predict labels',learning:'supervised',metric:'accuracy',validation:'cv',folds:3,seeds:[42],testFraction:.2,holdoutFraction:.2,splitStrategy:'independent',assetColumns:[],excludedColumns:[],model:'sklearn.tree.DecisionTreeClassifier',searchModels:false,exportModel:false,output:'py',policy:{augmentation:false,regularization:false,features:false,tuning:false,pretrained:false,ensemble:false}};
const source=`import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
p=argparse.ArgumentParser()
for name in ('train','validation','metadata','models','output','seed','config'): p.add_argument('--'+name)
a=p.parse_args()
m=json.loads(Path(a.metadata).read_text())
train=pd.read_csv(a.train); valid=pd.read_csv(a.validation)
assert 'label' not in valid.columns
model=DecisionTreeClassifier(max_depth=2,random_state=int(a.seed)).fit(train[m['features']],train['label'])
Path(a.output).write_text(json.dumps(model.predict(valid[m['features']]).tolist()))
`;
try {
  await runProcess(python,['-c',`import csv,sys
from pathlib import Path
root=Path(sys.argv[1])
for name,n,padding in [('million.csv',1000000,''),('wide-file.csv',150001,'z'*1000)]:
    with (root/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['x','y','label']+(['unused'] if padding else []))
        for i in range(n): w.writerow([i,i%17,'a' if i<n//2 else 'b']+([padding] if padding else []))
`,root],{timeout:120_000});
  const config=join(root,'task.json');await writeFile(config,JSON.stringify(task));
  const benchmark=await runProcess(python,['-c',`import json,sys,time,resource
sys.path.insert(0,sys.argv[1])
from prepare import prepare,materialize
from pathlib import Path
root=Path(sys.argv[2]);task=json.loads((root/'task.json').read_text());start=time.monotonic()
meta=prepare(task,root/'million'/'evaluation')
assert meta['rows']==1000000
protocol=json.loads((root/'million'/'evaluation'/'protocol.json').read_text())
f=protocol['evaluations'][0];expected=f['validation'];del protocol
truth=materialize(root/'million'/'evaluation',root/'materialized',f['id'])
assert len(truth['targets'])==len(expected)
assert (root/'materialized'/'validation.csv').open().readline().strip()=='x,y'
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
assert peak<512*1024**2,peak
print(json.dumps({'rows':meta['rows'],'peakMiB':round(peak/1024**2,1),'seconds':round(time.monotonic()-start,1),'estimateMiB':round(meta['resources']['preparationBytes']/1024**2,1)}))
`,join(process.cwd(),'local'),root],{timeout:600_000});
  console.log('Million-row preparation + streamed fold: '+benchmark.output.trim());
  const workspace=join(root,'million','source');await mkdir(workspace);await writeFile(join(workspace,'train.py'),source);
  const run={id:'million',task};
  const result=await evaluateCandidate(root,run,workspace,{deadline:null,candidateId:'tree'});
  assert.equal(result.evaluations.length,3);assert.ok(result.score>.99);assert.equal(result.evaluations[0].trainingRows+result.evaluations[0].validationRows,800000);
  console.log(`Million-row decision tree: all 3 CPU folds passed, accuracy ${result.score.toFixed(5)}.`);
  await writeFile(join(workspace,'train.py'),'from sklearn.decomposition import KernelPCA\nmodel=KernelPCA()');
  await assert.rejects(evaluateCandidate(root,run,workspace,{deadline:null,candidateId:'kernel'}),/pairwise matrices/);
  console.log('Same dataset: dense KernelPCA rejected by model-aware memory preflight before execution.');
  // A large source with excluded text must be scanned, not buffered or counted as a model feature.
  const wideTask={...task,dataset:join(root,'wide-file.csv'),excludedColumns:['unused'],validation:'holdout'};
  assert.ok((await stat(wideTask.dataset)).size>128*1024**2);
  const wide=await inspectDataset(wideTask);assert.equal(wide.rows,150001);assert.deepEqual(wide.features,['x','y']);
  console.log('CSV over 128 MiB accepted with excluded text; 150,001 rows and two modeled features.');
  // Model width and protocol complexity must affect limits independently of row count.
  await writeFile(join(workspace,'train.py'),source);
  await runProcess(python,['-c',String.raw`import json,sys,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from prepare import estimate_resources,SplitRows,check_preparation_memory,memory_limit
root=Path(sys.argv[2]);m=json.loads((root/'million'/'evaluation'/'manifest.json').read_text());t=json.loads((root/'task.json').read_text())
small=estimate_resources(m,t)
wide={**m,'features':['f'+str(i) for i in range(200)]}
assert small['estimatedFitBytes']<small['limitBytes']<estimate_resources(wide,t)['estimatedFitBytes']
source=root/'million'/'source'
(source/'train.py').write_text('import pandas as pd\nreader=pd.read_csv("train.csv",chunksize=5000)\nmodel.partial_fit(batch)')
assert estimate_resources(wide,t,source)['estimatedFitBytes']<small['limitBytes']
assert estimate_resources({**wide,'taskType':'clustering'},t,source)['estimatedScoringBytes']>small['limitBytes']
(source/'train.py').write_text('from sklearn.decomposition import KernelPCA as KP\nmodel=KP()')
assert estimate_resources(m,t,source)['estimatedFitBytes']>small['limitBytes']
os.environ['HELIX_TRAIN_MEMORY_MB']='2048'
assert memory_limit('TRAIN')==2048*1024**2
r=SplitRows([]);r.count=1000000
check_preparation_memory(t,r)
try: check_preparation_memory({**t,'folds':10,'seeds':[1,2,3,4,5]},r)
except ValueError as e: assert 'fewer folds/seeds' in str(e)
else: raise AssertionError('oversized split metadata accepted')
print('Width, incremental fitting, trusted scoring, model aliases and fold/seed count affect memory admission.')
`,join(process.cwd(),'local'),root],{timeout:30_000}).then(value=>console.log(value.output.trim()));
} finally {await cleanupContainers(root).catch(()=>{});await rm(root,{recursive:true,force:true});}
