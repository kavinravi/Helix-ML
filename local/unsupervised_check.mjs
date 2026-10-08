// Real CPU fits, trusted metrics, fresh-process export reload, and unlabeled final outputs.
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm, cp, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { prepareEvaluation, evaluateCandidate, evaluateFinal } from './evaluation.mjs';
import { cleanupContainers, container, IMAGE } from './runtime.mjs';
import { finishBundle } from './export.mjs';
import { callTool } from './tools.mjs';
import { runProcess } from './process.mjs';
const root = await mkdtemp(join(tmpdir(), 'helix-unsupervised-'));
const source = `import argparse, csv, json, joblib
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from helix_features import read_features, preprocessor, dense_features
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
p = argparse.ArgumentParser()
for flag in ('train', 'validation', 'metadata', 'models', 'output', 'seed', 'config'): p.add_argument('--' + flag)
a = p.parse_args()
c = json.loads(Path(a.config).read_text()); m = json.loads(Path(a.metadata).read_text())
assert m['target'] == ''
def matrix(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        assert 'label' not in reader.fieldnames and '' not in reader.fieldnames
    return read_features(path, m['features'])
train, valid = matrix(a.train), matrix(a.validation)
if c['learning'] == 'clustering':
    estimator = KMeans(n_clusters=3, n_init=10, random_state=int(a.seed))
else:
    components = c['varianceTarget'] if c['reductionMode'] == 'variance' else c['dimensions']
    estimator = PCA(n_components=None if components == 1 and c['reductionMode'] == 'variance' else components, svd_solver='full')
encoding = preprocessor(train)
model = (make_pipeline(encoding, estimator) if c['learning']=='clustering' else make_pipeline(encoding, FunctionTransformer(dense_features), estimator)).fit(train)
result = model.predict(valid) if c['learning'] == 'clustering' else model.transform(valid)
Path(a.output).write_text(json.dumps(result.tolist()))
if c['exportModel']:
    folder = Path(a.output).parent / 'model'; folder.mkdir()
    joblib.dump(model, folder / 'model.joblib')
    (folder / 'model_manifest.json').write_text(json.dumps({'format':'joblib','files':['model.joblib']}))
`;
const predict = `import argparse, csv, json, joblib
from pathlib import Path
p = argparse.ArgumentParser()
for flag in ('input','metadata','config','model-dir','models','output'): p.add_argument('--'+flag)
a=p.parse_args(); m=json.loads(Path(a.metadata).read_text()); c=json.loads(Path(a.config).read_text())
assert not Path('/data/train.csv').exists()
from helix_features import read_features
rows=read_features(a.input,m['features'])
model=joblib.load(Path(a.model_dir)/'model.joblib')
Path(a.output).write_text(json.dumps((model.predict(rows) if c['learning']=='clustering' else model.transform(rows)).tolist()))
`;
try {
  const dataset = join(root, 'data.csv');
  await writeFile(dataset, 'a,b,c,d,label\n' + Array.from({ length: 120 }, (_, i) => {
    const group = i % 3, noise = Math.sin(i * 1.9);
    return `${group * 8 + noise},${group * 6 + Math.cos(i)},${Math.sin(i)},${group * 7 + noise * .2},group${group}`;
  }).join('\n'));
  const mixed = join(root,'mixed.csv'), textOnly=join(root,'text.csv');
  await writeFile(mixed, 'a,b,category,notes,label\n'+Array.from({length:120},(_,i)=>`${i%3*8+Math.sin(i)},${Math.cos(i)},${['red','green','blue'][i%3]},${['apple fruit orchard','forest trees leaves','ocean wave water'][i%3]} record${i},label${i%3}`).join('\n'));
  await writeFile(textOnly, 'notes\n'+Array.from({length:120},(_,i)=>`${['apple fruit orchard','forest trees leaves','ocean wave water'][i%3]} record${i}`).join('\n'));
  const image = (await runProcess('docker', ['image','inspect',IMAGE,'--format','{{.Id}}'])).output.trim();
  for (const mode of process.env.HELIX_TEST_UMAP ? ['umap'] : ['silhouette','davies_bouldin','dimensions','variance','all-variance','mixed-silhouette','mixed-davies_bouldin','mixed-dimensions','mixed-variance','text-dimensions']) {
    const variant=mode.replace(/^(mixed|text)-/,'');
    const learning = variant.includes('variance') || ['dimensions','umap'].includes(variant) ? 'reduction' : 'clustering';
    const task = { dataset:mode.startsWith('mixed-')?mixed:mode.startsWith('text-')?textOnly:dataset, target:'', learning, metric: learning === 'clustering' ? variant : 'trustworthiness', dimensions:2, reductionMode: mode.includes('variance') ? 'variance' : 'dimensions', varianceTarget: mode === 'all-variance' ? 1 : .8, excludedColumns:mode.startsWith('text-')?[]:['label'], validation:'cv', folds:3, seeds:[42], splitStrategy:'independent', testFraction: mode === 'dimensions' ? .2 : 0, holdoutFraction:.2, assetColumns:[], searchModels:false, model:learning === 'clustering' ? 'sklearn.cluster.KMeans' : 'sklearn.decomposition.PCA', output:'py', exportModel:true, exportFormat:'joblib', policy:{augmentation:false,features:false,regularization:false,tuning:false,pretrained:false,ensemble:false} };
    const run = { id: mode, task, best:'candidate', trials:[] };
    const workspace = join(root, run.id, 'source'); await mkdir(workspace, { recursive:true });
    await writeFile(join(workspace,'train.py'), mode === 'umap' ? source.replace('from sklearn.decomposition import PCA','from sklearn.decomposition import PCA\nfrom umap import UMAP').replace("estimator = PCA(n_components=None if components == 1 and c['reductionMode'] == 'variance' else components, svd_solver='full')", "estimator = UMAP(n_components=c['dimensions'], n_neighbors=10, n_epochs=30, random_state=int(a.seed), transform_seed=int(a.seed), n_jobs=1)") : source); await writeFile(join(workspace,'predict.py'), predict);
    if (mode === 'umap') {
      task.validation='holdout';
      const packages=join(root,run.id,'dependencies','candidate');await mkdir(packages,{recursive:true});
      const runFile=join(root,run.id,'run.json');await writeFile(runFile,JSON.stringify(run));
      const installContext={root,runId:run.id,runFile,packages,deadline:null,image,user:`${process.getuid()}:${process.getgid()}`};
      await callTool('install_package',{name:'umap-learn',version:'0.5.8'},installContext);
      assert.match(await callTool('install_package',{name:'umap-learn',version:'0.5.8'},installContext),/already installed/);
    }
    const metadata = await prepareEvaluation(root, run, { deadline:null });
    assert.equal(metadata.target,''); assert.equal(metadata.taskType,learning); assert.equal(metadata.features.length,mode.startsWith('text-')?1:4); assert.equal(metadata.featureEncoding,'mixed-v1');
    if (mode === 'silhouette') {
      delete metadata.featureEncoding; // A saved numeric-only run must remain usable after the update.
      await writeFile(join(root,run.id,'evaluation','manifest.json'),JSON.stringify(metadata));
    }
    const options = { image, deadline:null, candidateId:'candidate', ...(mode === 'umap' ? {packages:join(root,run.id,'dependencies','candidate')} : {}) };
    if (mode === 'silhouette') {
      const outside=join(root,'untouched.txt'); await writeFile(outside,'untouched');
      await symlink(outside,join(workspace,'helix_features.py'));
      await assert.rejects(evaluateCandidate(root,run,workspace,options),/symlinks/);
      assert.equal(await readFile(outside,'utf8'),'untouched');
      await rm(join(workspace,'helix_features.py'));
    }
    const result = await evaluateCandidate(root,run,workspace,options);
    assert.equal(result.evaluations.length,mode === 'umap' ? 1 : 3); assert.equal(result.baseline,null); assert.ok(Number.isFinite(result.score));
    if (mode === 'silhouette') assert.ok(result.score > .35);
    if (learning === 'reduction') assert.ok(result.score >= (mode.includes('-') && mode !== 'all-variance' ? .5 : .75) && result.score <= 1);
    if (mode.includes('variance')) assert.ok(result.evaluations.every(e => e.cumulativeVariance + 1e-12 >= task.varianceTarget));
    const final = await evaluateFinal(root,run,workspace,options);
    assert.deepEqual(await evaluateFinal(root,run,workspace,options),final);
    assert.equal(final.projection.rows,mode === 'dimensions' ? 24 : 120);
    assert.equal(final.projection.scope,mode === 'dimensions' ? 'Test rows' : 'Full dataset');
    if (mode !== 'dimensions') assert.equal(final.score,null);
    if (mode === 'all-variance') assert.equal(final.projection.dimensions,4);
    run.projection=final.projection; run.score=result.score; run.testScore=final.score;
    await mkdir(join(root,run.id,'data')); await cp(join(root,run.id,'evaluation','manifest.json'),join(root,run.id,'data','manifest.json'));
    await mkdir(join(root,run.id,'final','source'),{recursive:true}); await cp(workspace,join(root,run.id,'final','source'),{recursive:true});
    await mkdir(join(root,run.id,'dependencies','candidate'),{recursive:true});
    await finishBundle(root,run,{root,runId:run.id,deadline:null,image,user:`${process.getuid()}:${process.getgid()}`,models:join(root,run.id,'models')});
    assert.ok((await readFile(join(root,run.id,'final','source','helix_features.py'),'utf8')).includes('def preprocessor'));
    const representation = (await readFile(join(root,run.id,'final','representation.csv'),'utf8')).trim().split('\n');
    assert.equal(representation.length,final.projection.rows+1);
    assert.equal(new Set(representation.slice(1).map(row=>row.split(',')[0])).size,final.projection.rows);
    console.log(`${mode}: ${result.evaluations.length} validation fits, independent score ${result.score.toFixed(4)}, final representation and model reload passed`);
    if (mode === 'dimensions') {
      const followup = { ...run,id:'followup',followup:{parentId:run.id} }; await mkdir(join(root,followup.id));
      await prepareEvaluation(root,followup,{deadline:null});
      const continued=await evaluateFinal(root,followup,workspace,options);
      assert.equal(continued.score,null); assert.equal(continued.projection.scope,'Training pool'); assert.equal(continued.projection.rows,96);
      const testRows=JSON.parse(await readFile(join(root,run.id,'evaluation','protocol.json'),'utf8')).test;
      assert.ok(continued.projection.points.every(p=>!testRows.includes(p.row)));
    }
  }
  if (!process.env.HELIX_TEST_UMAP) {
  const encodingChecks=String.raw`import os, sys
sys.path.insert(0,'/harness')
import numpy as np
import pandas as pd
from scipy import sparse
from helix_features import preprocessor, dense_features, read_features
train=pd.DataFrame({'number':['1','3',''], 'category':['red','blue','red'], 'text':['apple fruit orchard','ocean wave water','apple fruit leaves'], 'empty':['','','']})
valid=pd.DataFrame({'number':['100'], 'category':['never-seen'], 'text':['unseenword fruit'], 'empty':['new text']})
encoder=preprocessor(train); encoded=encoder.fit_transform(train); result=encoder.transform(valid)
names=encoder.get_feature_names_out().tolist()
assert sparse.issparse(encoded) and encoded.shape[1]>len(train.columns)
assert all('never-seen' not in name and 'unseenword' not in name for name in names)
assert not any(result[0,i] for i,name in enumerate(names) if name.startswith('f1__'))
assert result[0,0]>50 and result[0,-1]==0
assert encoder.named_transformers_['f0'].named_steps['simpleimputer'].statistics_[0]==2
assert 'unseenword' not in encoder.named_transformers_['f2'].vocabulary_
encoder.transform(valid); assert encoder.get_feature_names_out().tolist()==names
symbols=pd.DataFrame({'category':['!!!','???','!!!']})
assert preprocessor(symbols).fit_transform(symbols).shape==(3,2)
bad=pd.DataFrame({'number':['1','inf']})
try: preprocessor(bad).fit_transform(bad)
except ValueError as error: assert 'finite' in str(error)
else: raise AssertionError('infinite numeric value accepted')
os.environ['HELIX_TRAIN_MEMORY_MB']='256'
try: dense_features(sparse.csr_matrix((1000,10000)))
except ValueError as error: assert 'encoded features' in str(error)
else: raise AssertionError('oversized dense encoding accepted')
print('Encoding: train-only statistics/vocabulary, unknown categories, missing values, punctuation and expanded memory guard passed')
`;
  console.log((await container({root,runId:'encoding',deadline:null,image,user:`${process.getuid()}:${process.getgid()}`},['python','-c',encodingChecks],{mounts:[[join(process.cwd(),'local'),'/harness']]})).output.trim());
  // Scorer rejection checks run in the same trusted environment, without candidate packages.
  const data=join(root,'scorer-data');
  await runProcess(process.env.HELIX_PYTHON || 'python3',[join(process.cwd(),'local','prepare.py'),'--evaluation',join(root,'variance','evaluation'),'--output',data,'--fold','final']);
  const checks=`import json, sys\nfrom pathlib import Path\nsys.path.insert(0,'/harness')\nfrom unsupervised import evaluate\np=Path('/data'); m=json.loads((p/'manifest.json').read_text()); n=120\ndef reject(values):\n    try: evaluate(p,values,42)\n    except ValueError: return\n    raise AssertionError('invalid outputs accepted')\nreject([[0,0]]*n)\nreject([[float('nan'),0]]*n)\nreject([[0]]*n)\nprint('Invalid PCA geometry, nonfinite values and wrong dimensions rejected')\n`;
  console.log((await container({root,runId:'checks',deadline:null,image,user:`${process.getuid()}:${process.getgid()}`},['python','-c',checks],{mounts:[[join(process.cwd(),'local'),'/harness'],[data,'/data']]})).output.trim());
}
} finally { await cleanupContainers(root); await rm(root,{recursive:true,force:true}); }
