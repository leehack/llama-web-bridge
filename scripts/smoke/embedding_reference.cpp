// Independent upstream-only CPU reference, following llamadart-native's
// tools/embedding_smoke.cpp. See CONTRIBUTING.md for the exact source/build.
#include "llama.h"
#include "mtmd.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
  if (!ok) throw std::runtime_error(message);
}

static std::vector<float> run(llama_context * ctx, mtmd_context * projector,
    const std::string & prompt, const std::vector<const mtmd_bitmap *> & bitmaps) {
  llama_memory_clear(llama_get_memory(ctx), false);
  const auto * model = llama_get_model(ctx);
  require(llama_model_n_embd_inp(model) == 512 && llama_model_n_embd_out(model) == 768,
      "expected EmbeddingGemma-2 widths");
  if (bitmaps.empty()) {
    std::vector<llama_token> tokens(prompt.size() + 32);
    int count = llama_tokenize(llama_model_get_vocab(model), prompt.data(), prompt.size(),
        tokens.data(), tokens.size(), true, true);
    require(count > 0 && count <= int(tokens.size()), "text tokenize failed");
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
      batch.token[i] = tokens[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(status == 0, "text decode failed");
  } else {
    mtmd_input_text text{prompt.data(), prompt.size(), true, true};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>
        chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    require(mtmd_tokenize(projector, chunks.get(), &text, bitmaps.data(), bitmaps.size()) == 0,
        "media tokenize failed");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)>
        batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
    std::vector<std::vector<float>> media;
    llama_pos position = 0;
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
      const auto * chunk = mtmd_input_chunks_get(chunks.get(), c);
      size_t n = mtmd_input_chunk_get_n_tokens(chunk);
      const llama_token * tokens = nullptr;
      const float * embeddings = nullptr;
      require(n <= llama_n_ubatch(ctx) - size_t(position), "input exceeds one microbatch");
      if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
      } else {
        require(mtmd_encode_chunk(projector, chunk) == 0, "encode failed");
        const float * encoded = mtmd_get_output_embd(projector);
        require(encoded, "missing media output");
        media.emplace_back(encoded, encoded + n * 512);
        embeddings = media.back().data();
      }
      for (size_t i = 0; i < n; ++i, ++position) {
        int index = tokens ? llama_batch_ext_add_token(batch.get(), 0, tokens[i])
            : llama_batch_ext_add_embd(batch.get(), 0, {embeddings + i * 512, 1, 512});
        require(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &position)
            && llama_batch_ext_set_output_embd(batch.get(), index, true), "batch entry failed");
      }
    }
    require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "mixed process failed");
  }
  const float * result = llama_get_embeddings_seq(ctx, 0);
  require(result, "missing pooled output");
  return {result, result + 768};
}

int main(int argc, char ** argv) {
  if (argc != 4) return 2;
  llama_backend_init();
  try {
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 2048; cp.n_threads = cp.n_threads_batch = 4;
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_MEAN; cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context load failed");
    auto pp = mtmd_context_params_default(); pp.use_gpu = false; pp.warmup = false; pp.n_threads = 4;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> projector(mtmd_init_from_file(argv[2], model.get(), pp), mtmd_free);
    require(bool(projector), "projector load failed");
    require(mtmd_support_vision(projector.get()) && mtmd_support_audio(projector.get())
        && mtmd_get_audio_sample_rate(projector.get()) == 16000, "modality probe failed");
    std::vector<unsigned char> pixels(224*224*3, 0);
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> black(mtmd_bitmap_init(224,224,pixels.data()), mtmd_bitmap_free);
    for (size_t i=0;i<pixels.size();++i) pixels[i]=((i/3/224/16+i/3%224/16)%2)?255:0;
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> checker(mtmd_bitmap_init(224,224,pixels.data()), mtmd_bitmap_free);
    std::vector<float> samples(16000, 0);
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> silence(mtmd_bitmap_init_from_audio(samples.size(),samples.data()),mtmd_bitmap_free);
    for(size_t i=0;i<samples.size();++i) samples[i]=0.2f*std::sin(2*3.141592653589793*440*i/16000);
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> tone(mtmd_bitmap_init_from_audio(samples.size(),samples.data()),mtmd_bitmap_free);
    std::ofstream output(argv[3]); output << std::setprecision(9) << "{";
    const auto emit = [&](const char * name, const std::string & text, const std::vector<const mtmd_bitmap *> & media, bool first=false) {
      const auto raw = run(ctx.get(),projector.get(),text,media);
      double norm=0; for(float v:raw) { require(std::isfinite(v),"nonfinite output"); norm+=double(v)*v; }
      require(norm>0,"zero output"); norm=std::sqrt(norm);
      output << (first?"":",") << "\"" << name << "\":{\"raw\":[";
      for(size_t i=0;i<raw.size();++i) output << (i?",":"") << raw[i];
      output << "],\"normalized\":[";
      for(size_t i=0;i<raw.size();++i) output << (i?",":"") << float(raw[i]/norm);
      output << "]}";
    };
    const std::string marker=mtmd_default_marker();
    emit("text","task: search result | title: Paris | text: The Eiffel Tower is in Paris.",{},true);
    for(const auto & pair:std::vector<std::pair<const char *,const mtmd_bitmap *>>{
        {"black",black.get()},{"checker",checker.get()},{"silence",silence.get()},{"tone",tone.get()}})
      emit(pair.first,"task: search result | text: "+marker,{pair.second});
    emit("interleaved","task: search result | text: A "+marker+" B\n雪"+marker+" C "+marker,
        {black.get(),tone.get(),checker.get()});
    output << "}\n";
    require(bool(output), "reference output write failed");
  } catch(const std::exception & e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
  llama_backend_free();
}
