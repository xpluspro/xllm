/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "attention.h"

#include <glog/logging.h>

#include <sstream>
#include <vector>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

std::string TensorBrief(const torch::Tensor& t) {
  if (!t.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << "sizes=" << t.sizes()
      << ", dtype=" << static_cast<int>(t.scalar_type())
      << ", device=" << t.device() << ", contiguous=" << t.is_contiguous();
  return oss.str();
}

void ChunkedPrefillByPagedDecode(const torch::Tensor& query,
                                 torch::Tensor& output,
                                 const torch::Tensor& k_cache,
                                 const torch::Tensor& v_cache,
                                 const AttentionMetadata& attn_metadata,
                                 float scale,
                                 int64_t num_heads,
                                 int64_t head_size) {
  CHECK(attn_metadata.q_seq_lens.defined()) << "q_seq_lens is required";
  CHECK(attn_metadata.kv_seq_lens_host.defined())
      << "kv_seq_lens_host is required";
  CHECK(attn_metadata.block_table.defined()) << "block_table is required";

  auto q_seq_lens_cpu =
      attn_metadata.q_seq_lens.to(torch::kCPU).to(torch::kInt).contiguous();
  auto kv_seq_lens_cpu = attn_metadata.kv_seq_lens_host.to(torch::kCPU)
                             .to(torch::kInt)
                             .contiguous();

  CHECK_EQ(q_seq_lens_cpu.dim(), 1) << "q_seq_lens must be 1D";
  CHECK_EQ(kv_seq_lens_cpu.dim(), 1) << "kv_seq_lens_host must be 1D";
  CHECK_EQ(q_seq_lens_cpu.size(0), kv_seq_lens_cpu.size(0))
      << "q_seq_lens and kv_seq_lens_host size mismatch";
  CHECK_EQ(attn_metadata.block_table.size(0), q_seq_lens_cpu.size(0))
      << "block_table batch size mismatch";

  auto q_accessor = q_seq_lens_cpu.accessor<int, 1>();
  auto kv_accessor = kv_seq_lens_cpu.accessor<int, 1>();

  int64_t q_offset = 0;
  for (int64_t seq_idx = 0; seq_idx < q_seq_lens_cpu.size(0); ++seq_idx) {
    const int32_t q_len = q_accessor[seq_idx];
    const int32_t kv_len = kv_accessor[seq_idx];
    CHECK_GE(q_len, 0) << "q_len must be non-negative";
    CHECK_GE(kv_len, q_len) << "kv_len must be >= q_len";
    if (q_len == 0) {
      continue;
    }

    CHECK_LE(q_offset + q_len, query.size(0)) << "query slice out of range";

    auto query_slice =
        query.narrow(0, q_offset, q_len).view({q_len, 1, num_heads, head_size});
    auto output_slice = output.narrow(0, q_offset, q_len)
                            .view({q_len, 1, num_heads, head_size});

    auto block_row = attn_metadata.block_table.narrow(0, seq_idx, 1);
    auto block_table = block_row.repeat({q_len, 1});

    std::vector<int32_t> seq_lens_vec;
    seq_lens_vec.reserve(q_len);
    const int32_t prefix_len = kv_len - q_len;
    for (int32_t i = 0; i < q_len; ++i) {
      seq_lens_vec.push_back(prefix_len + i + 1);
    }
    auto seq_lens = torch::tensor(seq_lens_vec, torch::kInt);

    xllm::kernel::npu::batch_decode(query_slice,
                                    k_cache,
                                    v_cache,
                                    scale,
                                    block_table,
                                    seq_lens,
                                    output_slice);
    q_offset += q_len;
  }

  CHECK_EQ(q_offset, query.size(0)) << "q_offset mismatch with query tokens";
}

}  // namespace

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  VLOG(1) << "[NPU prefill] is_prefill=" << attn_metadata.is_prefill
          << ", is_chunked_prefill=" << attn_metadata.is_chunked_prefill
          << ", max_query_len=" << attn_metadata.max_query_len
          << ", max_seq_len=" << attn_metadata.max_seq_len
          << ", total_kv_len=" << attn_metadata.total_kv_len << ", query{"
          << TensorBrief(query) << "}"
          << ", output{" << TensorBrief(output) << "}"
          << ", attn_mask{" << TensorBrief(attn_metadata.attn_mask) << "}"
          << ", kv_seq_lens_host{"
          << TensorBrief(attn_metadata.kv_seq_lens_host) << "}";

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

    VLOG(1) << "[NPU prefill] prefill path key{" << TensorBrief(key)
            << "}, value{" << TensorBrief(value) << "}";

    xllm::kernel::npu::batch_prefill(query,
                                     key,
                                     value,
                                     attn_metadata.attn_mask,
                                     attn_metadata.kv_seq_lens_host,
                                     scale_,
                                     output);
  } else if (attn_metadata.is_chunked_prefill) {
    CHECK(v_cache.has_value() && v_cache->defined())
        << "v_cache is required for chunked prefill";

    bool has_append_kv = false;
    if (attn_metadata.q_seq_lens.defined() &&
        attn_metadata.kv_seq_lens_host.defined()) {
      auto q_seq_lens_cpu =
          attn_metadata.q_seq_lens.to(torch::kCPU).to(torch::kInt).contiguous();
      auto kv_seq_lens_cpu = attn_metadata.kv_seq_lens_host.to(torch::kCPU)
                                 .to(torch::kInt)
                                 .contiguous();
      CHECK_EQ(q_seq_lens_cpu.size(0), kv_seq_lens_cpu.size(0))
          << "q_seq_lens and kv_seq_lens_host size mismatch";
      auto q_accessor = q_seq_lens_cpu.accessor<int, 1>();
      auto kv_accessor = kv_seq_lens_cpu.accessor<int, 1>();
      for (int64_t i = 0; i < q_seq_lens_cpu.size(0); ++i) {
        if (q_accessor[i] != kv_accessor[i]) {
          has_append_kv = true;
          break;
        }
      }
    }

    if (has_append_kv) {
      VLOG(1) << "[NPU prefill] chunked prefill append mode uses paged decode "
                 "fallback"
              << ", k_cache{" << TensorBrief(k_cache) << "}, v_cache{"
              << TensorBrief(v_cache.value()) << "}";
      ChunkedPrefillByPagedDecode(query,
                                  output,
                                  k_cache,
                                  v_cache.value(),
                                  attn_metadata,
                                  scale_,
                                  num_heads_,
                                  head_size_);
    } else {
      key = key.view({-1, num_kv_heads_, head_size_});
      value = value.view({-1, num_kv_heads_, head_size_});

      VLOG(1) << "[NPU prefill] chunked prefill dense mode k_cache{"
              << TensorBrief(k_cache) << "}, v_cache{"
              << TensorBrief(v_cache.value_or(torch::Tensor())) << "}, key{"
              << TensorBrief(key) << "}, value{" << TensorBrief(value) << "}";

      xllm::kernel::npu::batch_prefill(query,
                                       key,
                                       value,
                                       attn_metadata.attn_mask,
                                       attn_metadata.kv_seq_lens_host,
                                       scale_,
                                       output);
    }
  }
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (attn_metadata.paged_attention_tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    xllm::kernel::npu::batch_decode_acl_graph(
        query,
        k_cache,
        v_cache.value_or(torch::Tensor()),
        scale_,
        attn_metadata.block_table,
        kv_seq_lens,
        attn_metadata.paged_attention_tiling_data,
        output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    attn_metadata.block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
