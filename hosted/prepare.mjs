// Run with a Railway project token and environment ID. Contains no user credentials.
import {Sandbox} from 'railway';
import {readFile} from 'node:fs/promises';
const sandbox=await Sandbox.create({idleTimeoutMinutes:5});
try {
  await sandbox.files.write('/opt/helix/runtime/Dockerfile',await readFile(new URL('../local/Dockerfile',import.meta.url)));
  const result=await sandbox.exec('docker build -t helix-ml:local /opt/helix/runtime',{timeoutSec:600,onStdout:c=>process.stdout.write(c),onStderr:c=>process.stderr.write(c)});
  if(result.exitCode!==0)throw new Error('Training image build failed.');
  const checkpoint=await sandbox.checkpoint('helix-base-v1');
  console.log(`Training runtime ready: ${checkpoint.key}`);
} finally {await sandbox.destroy();}
