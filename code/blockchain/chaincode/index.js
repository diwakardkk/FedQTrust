'use strict';
const {Contract} = require('fabric-contract-api');
class FedQTrust extends Contract {
  async Ping() { return 'pong'; }
  async ProtocolVersion() { return 'fedqtrust-manuscript-v2'; }
  async owner(ctx,run) {
    const key=ctx.stub.createCompositeKey('owner',[run]);
    const owner=(await ctx.stub.getState(key)).toString();
    const id=ctx.clientIdentity.getID();
    if (owner && owner!==id) throw new Error('Run owner mismatch');
    if (!owner) await ctx.stub.putState(key,Buffer.from(id));
  }
  async RegisterSigner(ctx,run,signer,publicKey,scheme='ML-DSA-65') {
    await this.owner(ctx,run);
    const key=ctx.stub.createCompositeKey('signer',[run,signer]);
    const previous=(await ctx.stub.getState(key)).toString();
    const record=JSON.stringify({publicKey,scheme});
    if(previous && previous!==record) throw new Error('Signer key replacement denied');
    await ctx.stub.putState(key,Buffer.from(record));
  }
  async VerifyRecord(ctx,run,signer,digest,signature) {
    await this.owner(ctx,run);
    const raw=(await ctx.stub.getState(ctx.stub.createCompositeKey('signer',[run,signer]))).toString();
    if(!raw) throw new Error('Unregistered signer');
    const {publicKey,scheme}=JSON.parse(raw);
    let valid=false;
    if(scheme==='ML-DSA-65') {
      const {ml_dsa65}=await import('@noble/post-quantum/ml-dsa.js');
      valid=ml_dsa65.verify(Buffer.from(signature,'hex'),Buffer.from(digest,'hex'),Buffer.from(publicKey,'hex'));
    } else if(scheme==='ECDSA-P256') {
      const crypto=require('node:crypto');
      const key=crypto.createPublicKey({key:Buffer.from(publicKey,'hex'),format:'der',type:'spki'});
      valid=crypto.verify('sha256',Buffer.from(digest,'hex'),key,Buffer.from(signature,'hex'));
    }
    if(!valid) throw new Error('Invalid record signature');
    await ctx.stub.putState(ctx.stub.createCompositeKey('verified',[run,signer,digest]),Buffer.from(ctx.stub.getTxID()));
  }
  async SetTrust(ctx, run, client, value, digest='') {
    await this.owner(ctx,run);
    const trustKey=ctx.stub.createCompositeKey('trust',[run,client]);
    const prior=await ctx.stub.getState(trustKey);
    if(prior.length) {
      if(!digest) throw new Error('Verified client update required');
      const proof=await ctx.stub.getState(ctx.stub.createCompositeKey('verified',[run,client,digest]));
      const usedKey=ctx.stub.createCompositeKey('used',[run,client,digest]);
      if(!proof.length || (await ctx.stub.getState(usedKey)).length) throw new Error('Missing or replayed update proof');
      await ctx.stub.putState(usedKey,Buffer.from('1'));
    }
    const score=Number(value);
    if (!Number.isFinite(score)||score<0||score>1) throw new Error('Invalid trust score');
    await ctx.stub.putState(ctx.stub.createCompositeKey('trust',[run,client]),Buffer.from(String(score)));
  }
  async GetTrust(ctx,run,client) {
    return (await ctx.stub.getState(ctx.stub.createCompositeKey('trust',[run,client]))).toString();
  }
  async LogSelection(ctx,run,record) {
    await this.owner(ctx,run);
    JSON.parse(record);
    await ctx.stub.putState(ctx.stub.createCompositeKey('selection',[run,ctx.stub.getTxID()]),Buffer.from(record));
  }
}
module.exports.contracts=[FedQTrust];
