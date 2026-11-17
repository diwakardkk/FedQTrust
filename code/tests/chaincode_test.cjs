// Test contract logic with an in-memory ledger; cryptography uses real Node ECDSA.
const assert=require('node:assert/strict');
const Module=require('node:module');
const load=Module._load;
Module._load=function(name,...rest){return name==='fabric-contract-api'?{Contract:class{}}:load.call(this,name,...rest);};
const Contract=require('../blockchain/chaincode').contracts[0];
const crypto=require('node:crypto');
(async()=>{
 const contract=new Contract();const state=new Map();
 const ctx={clientIdentity:{getID:()=> 'owner'},stub:{
   createCompositeKey:(type,parts)=>JSON.stringify([type,...parts]),
   getState:async k=>state.get(k)||Buffer.alloc(0),
   putState:async(k,v)=>{state.set(k,v);},getTxID:()=> 'tx1'}};
 const keys=crypto.generateKeyPairSync('ec',{namedCurve:'prime256v1'});
 const pub=keys.publicKey.export({format:'der',type:'spki'}).toString('hex');
 await contract.RegisterSigner(ctx,'run','0',pub,'ECDSA-P256');
 await contract.SetTrust(ctx,'run','0','.65');
 await assert.rejects(()=>contract.SetTrust(ctx,'run','0','.7'),/Verified/);
 const digest=crypto.createHash('sha256').update('client update').digest();
 const signature=crypto.sign('sha256',digest,keys.privateKey).toString('hex');
 await contract.VerifyRecord(ctx,'run','0',digest.toString('hex'),signature);
 await contract.SetTrust(ctx,'run','0','.7',digest.toString('hex'));
 assert.equal(await contract.GetTrust(ctx,'run','0'),'0.7');
 await assert.rejects(()=>contract.SetTrust(ctx,'run','0','.7',digest.toString('hex')),/replayed/);
 await assert.rejects(()=>contract.VerifyRecord(ctx,'run','0','00'.repeat(32),signature),/Invalid/);
 ctx.clientIdentity.getID=()=> 'intruder';
 await assert.rejects(()=>contract.SetTrust(ctx,'run','0','.9'),/owner/);
 console.log('Chaincode identity, signature, proof and replay checks passed');
})().catch(e=>{console.error(e);process.exit(1);});
