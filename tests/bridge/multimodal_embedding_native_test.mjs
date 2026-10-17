import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { repoRoot } from './bridge_js_source.mjs';

// A deterministic implementation of the upstream API, not a reimplementation of
// the bridge algorithm. Its output is inspected only when llama_process runs.
// In particular add_embd keeps pointers, and encode reuses the same buffer.
const source = String.raw`
#include <cassert>
#include <cstdint>
#include <vector>
#include <string>
using llama_token = int;
using llama_pos = int;
enum llama_pooling_type { LLAMA_POOLING_TYPE_NONE, LLAMA_POOLING_TYPE_MEAN, LLAMA_POOLING_TYPE_RANK, LLAMA_POOLING_TYPE_CLS };
enum llama_process_type { LLAMA_PROCESS_TYPE_ENCODE, LLAMA_PROCESS_TYPE_DECODE };
struct llama_context { unsigned ctx=16, seq=16, batch=16, ubatch=16, seqmax=1; int pool=1; bool causal=false, extraction=false; };
struct llama_model { bool encoder=false, decoder=true; int out=3, inp=2; };
struct llama_vocab {};
struct mtmd_context { bool vision=true, audio=true; int rate=16000; };
struct mtmd_bitmap { bool audio=false; };
struct mtmd_input_chunk { int type; std::vector<int> tokens; size_t n; float value; };
struct mtmd_input_chunks { std::vector<mtmd_input_chunk> chunks; };
struct mtmd_input_text { const char * text; size_t length; bool special, parse; };
struct llama_embd { const float * data; size_t n_rows, n_embd; };
struct entry { int token; const float * data; int position=-1; bool output=false; };
struct llama_batch_ext { std::vector<entry> entries; };
struct llama_batch { int n_tokens; int * token; int * pos; int * n_seq_id; int ** seq_id; char * logits; };
constexpr int MTMD_INPUT_CHUNK_TYPE_TEXT = 0;
static int fail=0, encoded=0, processed=0, live_chunks=0, live_batches=0;
static float reused[4], pooled[3]={1,2,3};
static bool empty=false;
static bool huge=false;
static bool g_cancel_requested=false;
int llama_model_n_embd_out(llama_model * m) { return m->out; }
int llama_model_n_embd_inp(llama_model * m) { return m->inp; }
bool llama_model_has_encoder(llama_model * m) { return m->encoder; }
bool llama_model_has_decoder(llama_model * m) { return m->decoder; }
int llama_pooling_type(llama_context * c) { return c->pool; }
bool llama_get_causal_attn(llama_context * c) { return c->causal; }
bool mtmd_bitmap_is_audio(const mtmd_bitmap * b) { return b->audio; }
bool mtmd_support_audio(mtmd_context * p) { return p->audio; }
bool mtmd_support_vision(mtmd_context * p) { return p->vision; }
int mtmd_get_audio_sample_rate(mtmd_context * p) { return p->rate; }
mtmd_input_chunks * mtmd_input_chunks_init() { if(fail==1) return nullptr; ++live_chunks; return new mtmd_input_chunks; }
void mtmd_input_chunks_free(mtmd_input_chunks * c) { --live_chunks; delete c; }
int mtmd_tokenize(mtmd_context *, mtmd_input_chunks * c, const mtmd_input_text * t, const mtmd_bitmap * const *, size_t) {
  assert(t->special && t->parse && t->length==std::string(t->text).size());
  if(!empty) c->chunks={{0,{7},1,0}, {1,{},2,11}, {1,{},2,22}, {0,{8},1,0}};
  if(huge) c->chunks[1].n=SIZE_MAX;
  return fail==2?1:0;
}
size_t mtmd_input_chunks_size(mtmd_input_chunks * c) { return c->chunks.size(); }
const mtmd_input_chunk * mtmd_input_chunks_get(mtmd_input_chunks * c,size_t i) { return &c->chunks.at(i); }
size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * c) { return c->n; }
int mtmd_input_chunk_get_type(const mtmd_input_chunk * c) { return c->type; }
const int * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * c,size_t * n) { *n=c->n; return fail==3?nullptr:c->tokens.data(); }
int mtmd_encode_chunk(mtmd_context *,const mtmd_input_chunk * c) { ++encoded; for(auto & v:reused) v=c->value; return fail==4?1:0; }
const float * mtmd_get_output_embd(mtmd_context *) { return fail==5?nullptr:reused; }
llama_vocab * llama_model_get_vocab(llama_model *) { static llama_vocab v; return &v; }
int llama_tokenize(const llama_vocab *, const char *, size_t, int * out, size_t, bool special, bool parse) {
  assert(special && parse); if(fail==6) return 1; if(fail==18) return INT32_MIN; if(!out) return -2;
  if(fail==7) return -1; if(fail==16) return 3; out[0]=7; out[1]=8; return 2;
}
unsigned llama_n_ctx(llama_context * c) { return c->ctx; }
unsigned llama_n_ctx_seq(llama_context * c) { return c->seq; }
unsigned llama_n_batch(llama_context * c) { return c->batch; }
unsigned llama_n_ubatch(llama_context * c) { return c->ubatch; }
unsigned llama_n_seq_max(llama_context * c) { return c->seqmax; }
llama_batch_ext * llama_batch_ext_init(llama_context *) { if(fail==8) return nullptr; ++live_batches; return new llama_batch_ext; }
void llama_batch_ext_free(llama_batch_ext * b) { --live_batches; delete b; }
int llama_batch_ext_add_token(llama_batch_ext * b,int seq,int token) { assert(seq==0); if(fail==9) return -1; b->entries.push_back({token,nullptr}); return b->entries.size()-1; }
int llama_batch_ext_add_embd(llama_batch_ext * b,int seq,llama_embd e) { assert(seq==0 && e.n_rows==1 && e.n_embd==2); if(fail==10) return -1; b->entries.push_back({-1,e.data}); return b->entries.size()-1; }
bool llama_batch_ext_set_pos(llama_batch_ext * b,int i,const int * p) { b->entries.at(i).position=*p; return fail!=11; }
bool llama_batch_ext_set_output_embd(llama_batch_ext * b,int i,bool output) { b->entries.at(i).output=output; return fail!=12; }
void llama_synchronize(llama_context *) {}
void * llama_get_memory(llama_context *) { return nullptr; }
void llama_memory_clear(void *,bool) {}
void llama_set_embeddings(llama_context * c,bool value) { c->extraction=value; }
int llama_process(llama_context * c,llama_process_type type,llama_batch_ext * b) {
  ++processed; assert(c->extraction);
  assert(type==LLAMA_PROCESS_TYPE_DECODE || type==LLAMA_PROCESS_TYPE_ENCODE);
  for(size_t i=0;i<b->entries.size();++i) { assert(b->entries[i].position==int(i)); assert(b->entries[i].output); }
  assert(b->entries.front().token==7 && b->entries.back().token==8);
  if(b->entries.size()>2) { assert(b->entries.size()==6); assert(b->entries[1].data[0]==11 && b->entries[2].data[0]==11); assert(b->entries[3].data[0]==22 && b->entries[4].data[0]==22); }
  if(fail==15) g_cancel_requested=true;
  return (fail==13 || fail==15)?1:0;
}
const float * llama_get_embeddings_seq(llama_context *,int seq) { assert(seq==0); return fail==14?nullptr:pooled; }
llama_batch llama_batch_init(int n,int embd,int sequences) {
  assert(n>0 && embd==0 && sequences==1); ++live_batches;
  llama_batch b{n,new int[n],new int[n],new int[n],new int*[n],new char[n]};
  for(int i=0;i<n;++i) b.seq_id[i]=new int[1];
  if(fail==17) { delete[] b.token; b.token=nullptr; }
  return b;
}
void llama_batch_free(llama_batch b) {
  for(int i=0;i<b.n_tokens;++i) delete[] b.seq_id[i];
  delete[] b.token; delete[] b.pos; delete[] b.n_seq_id; delete[] b.seq_id; delete[] b.logits; --live_batches;
}
int llama_decode(llama_context * c,llama_batch b) {
  ++processed; assert(c->extraction);
  assert(b.n_tokens==2 && b.token[0]==7 && b.token[1]==8);
  for(int i=0;i<b.n_tokens;++i) assert(b.pos[i]==i && b.logits[i]==1 && b.n_seq_id[i]==1 && b.seq_id[i][0]==0);
  if(fail==15) g_cancel_requested=true;
  return fail==13 || fail==15?1:0;
}
int llama_encode(llama_context * c,llama_batch b) { return llama_decode(c,b); }
#ifndef MIXED_EMBEDDINGS_TEST
#define MIXED_EMBEDDINGS_TEST 1
#endif
#define LLAMADART_WEBGPU_HAS_MIXED_EMBEDDINGS MIXED_EMBEDDINGS_TEST
#include "src/llama_webgpu_embedding.h"
using namespace llamadart_webgpu_detail;
static struct { llama_context * ctx=nullptr; llama_model * model=nullptr; mtmd_context * mm_ctx=nullptr; } g_state;
static std::vector<struct pending_media> g_pending_media;
struct pending_media { const mtmd_bitmap * bitmap; };
static std::string g_last_embedding_json;
static bool g_generation_active=false, g_tts_active=false;
static std::vector<int> g_cached_prompt_tokens;
static int cleared=0;
static int normalized=0;
void clear_error() {}
bool ensure_loaded() { return g_state.ctx && g_state.model; }
void clear_pending_media() { ++cleared; g_pending_media.clear(); }
void normalize_embedding_inplace(std::vector<float> &) { ++normalized; }
namespace llamadart_webgpu_detail { std::string serialize_embedding_json(const std::vector<float> &) { return "[1,2,3]"; } }
#define EMSCRIPTEN_KEEPALIVE
#include "src/core/exports_embedding.inc"
int main() {
  llama_context c; llama_model m; mtmd_context p; mtmd_bitmap image, audio{true}; std::vector<float> out;
  int checks=0;
  const auto run=[&](int expected,bool media=true) {
    encoded=processed=0; out.clear();
    assert(embed_parts(&c,&m,&p,"x",media?std::vector<const mtmd_bitmap *>{&image,&audio}:std::vector<const mtmd_bitmap *>{},out)==expected);
    assert(live_batches==0 && live_chunks==0 && !c.extraction); ++checks;
  };
  assert(!supports_pooled_embeddings(nullptr,&m) && !supports_pooled_embeddings(&c,nullptr));
  run(3); assert(encoded==2 && processed==1 && out==std::vector<float>({1,2,3}));
  run(3,false); assert(encoded==0 && processed==1);
  for(auto field:{&llama_context::ctx,&llama_context::seq,&llama_context::batch,&llama_context::ubatch}) {
    c.*field=5; run(-23); assert(encoded==0 && processed==0); c.*field=16;
  }
  c.seqmax=0; run(-23); assert(processed==0); c.seqmax=1;
  for(int pool:{0,2}) { c.pool=pool; assert(!supports_pooled_embeddings(&c,&m)); run(-21); run(-21,false); } c.pool=1;
  m.encoder=true; assert(!supports_pooled_embeddings(&c,&m)); run(-21); run(-21,false); m.decoder=false; run(3,false); run(-21); m.encoder=false; m.decoder=true;
  m.out=0; run(-21); run(-21,false); m.out=3; m.inp=0; run(-25); assert(encoded==0 && processed==0); m.inp=2;
  c.causal=true; run(-21); c.causal=false; c.pool=3; run(-21); c.pool=1;
  p.vision=false; run(-24); p.vision=true; p.audio=false; run(-24); p.audio=true;
  p.rate=48000; run(-24); p.rate=16000;
  assert(embed_parts(&c,&m,nullptr,"x",{&image},out)==-22); ++checks;
  empty=true; run(-25); empty=false;
  huge=true; run(-23); assert(encoded==0 && processed==0); huge=false;
  for(int f=1;f<=14;++f) { fail=f; run(-25, f!=6 && f!=7); } fail=0;
  fail=16; run(-25,false); fail=0;
  for(int f:{13,14,17,18}) { fail=f; run(-25,false); } fail=0;
  c.ubatch=1; run(-23,false); assert(processed==0); c.ubatch=16;
  run(3); assert(processed==1); // recovery after every failed path
  g_state={&c,&m,&p};
  assert(std::string(llamadart_webgpu_embedding_capabilities_json())=="{\"apiVersion\":1,\"text\":true,\"image\":true,\"audio\":true}");
  g_state.mm_ctx=nullptr;
  assert(std::string(llamadart_webgpu_embedding_capabilities_json())=="{\"apiVersion\":1,\"text\":true,\"image\":false,\"audio\":false}");
  g_state.mm_ctx=&p;
  for(int condition=0;condition<6;++condition) {
    const int before=cleared;
    g_pending_media={{&image}};
    g_generation_active=condition==0; g_tts_active=condition==1;
    g_state.ctx=condition==2?nullptr:&c;
    fail=condition==4?15:0;
    assert(llamadart_webgpu_embed_parts_to_json(condition==3?nullptr:"x",1,condition==5?-1:1)==-25);
    assert(cleared==before+1 && g_pending_media.empty() && !g_cancel_requested);
    ++checks;
  }
  fail=0; g_state.ctx=&c; g_generation_active=g_tts_active=false;
  assert(llamadart_webgpu_embed_parts_to_json("x",0,1)==3 && !g_cancel_requested); ++checks;
  assert(normalized==0);
  assert(llamadart_webgpu_embed_parts_to_json("x",1,1)==3 && normalized==1); ++checks;
  std::printf("%d native cases passed\n",checks);
}
`;

test('mixed native batch guards, retained media, output width, one process and cleanup', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'embedding-native-'));
  try {
    writeFileSync(path.join(dir, 'llama.h'), '');
    writeFileSync(path.join(dir, 'mtmd.h'), '');
    writeFileSync(path.join(dir, 'test.cpp'), source);
    execFileSync(process.env.CXX || 'c++', ['-std=c++17', '-I', dir, '-I', repoRoot,
      path.join(dir, 'test.cpp'), '-o', path.join(dir, 'test')]);
    const output = execFileSync(path.join(dir, 'test'), { encoding: 'utf8' });
    assert.match(output, /55 native cases passed/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('pre-extended-batch core supports truthful text parts and rejects media without inference', () => {
  const dir=mkdtempSync(path.join(tmpdir(),'embedding-old-core-'));
  const oldMain=String.raw`
int main() {
  llama_context c; llama_model m; mtmd_context p; mtmd_bitmap image; std::vector<float> output;
  g_state={&c,&m,&p};
  assert(std::string(llamadart_webgpu_embedding_capabilities_json())=="{\"apiVersion\":1,\"text\":true,\"image\":false,\"audio\":false}");
  assert(llamadart_webgpu_embed_parts_to_json("x",1,1)==3 && processed==1);
  c.ubatch=1; processed=0;
  assert(llamadart_webgpu_embed_parts_to_json("x",1,1)==-23 && processed==0); c.ubatch=16;
  g_pending_media={{&image}};
  assert(llamadart_webgpu_embed_parts_to_json("x",1,1)==-20 && processed==0 && encoded==0 && g_pending_media.empty());
  fail=15;
  assert(llamadart_webgpu_embed_parts_to_json("x",1,1)==-25 && !g_cancel_requested && !c.extraction);
  fail=0;
  assert(llamadart_webgpu_embed_parts_to_json("x",0,1)==3 && !g_cancel_requested);
  assert(live_chunks==0 && live_batches==0);
  std::printf("6 old-core cases passed\n");
}
`;
  try {
    writeFileSync(path.join(dir,'llama.h'),'');
    writeFileSync(path.join(dir,'mtmd.h'),'');
    writeFileSync(path.join(dir,'test.cpp'),source.slice(0,source.indexOf('int main() {'))+oldMain);
    execFileSync(process.env.CXX || 'c++',['-std=c++17','-DMIXED_EMBEDDINGS_TEST=0','-I',dir,'-I',repoRoot,
      path.join(dir,'test.cpp'),'-o',path.join(dir,'test')]);
    assert.match(execFileSync(path.join(dir,'test'),{encoding:'utf8'}),/6 old-core cases passed/);
  } finally { rmSync(dir,{recursive:true,force:true}); }
});

test('media element multiplication rejects overflow in a bounded address space', () => {
  const dir=mkdtempSync(path.join(tmpdir(),'embedding-address-space-'));
  const bounded=source.slice(0,source.indexOf('int main() {'))
    .replace('#include "src/llama_webgpu_embedding.h"',
      '#undef SIZE_MAX\n#define SIZE_MAX 7\n#include "src/llama_webgpu_embedding.h"');
  try {
    writeFileSync(path.join(dir,'llama.h'),''); writeFileSync(path.join(dir,'mtmd.h'),'');
    writeFileSync(path.join(dir,'test.cpp'),bounded+String.raw`
int main() {
  llama_context c; llama_model m; m.inp=4; mtmd_context p; mtmd_bitmap image; std::vector<float> output;
  assert(llamadart_webgpu_detail::embed_parts(&c,&m,&p,"x",{&image},output)==-25);
  assert(encoded==1 && processed==0 && live_chunks==0 && live_batches==0);
}
`);
    execFileSync(process.env.CXX || 'c++',['-std=c++17','-I',dir,'-I',repoRoot,
      path.join(dir,'test.cpp'),'-o',path.join(dir,'test')]);
    execFileSync(path.join(dir,'test'));
  } finally { rmSync(dir,{recursive:true,force:true}); }
});
