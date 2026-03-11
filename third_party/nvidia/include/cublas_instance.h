#ifndef TRITON_CUBLAS_INSTANCE_H
#define TRITON_CUBLAS_INSTANCE_H

#include "cublas_types.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Information about a cuBLAS Lt matmul algorithm from heuristic search
struct AlgoHeuristicInfo {
  int index;                                // index in heuristic results list
  size_t workspace_size;                    // required workspace bytes
  float waves_count;                        // estimated wave count
  uint32_t tile_id;                         // configured tile ID (enum value)
  std::string tile_name;                    // configured tile name (e.g. "64x64")
  uint32_t stages_id;                       // configured stages ID
  uint32_t splitk_num;                      // configured split-K partitions
  uint32_t reduction_scheme;                // configured reduction scheme
  uint32_t cta_swizzling;                   // configured CTA swizzling
  uint32_t custom_option;                   // configured custom option value
  uint32_t custom_option_max;               // max custom option value
  std::vector<std::string> supported_tiles; // all supported tile names
  std::vector<uint32_t> supported_stages;   // all supported stage IDs
};

class CublasLtInstance {
private:
  // A validated algorithm entry: the algo object itself plus metadata.
  struct ValidatedAlgo {
    cublasLtMatmulAlgo_t algo;
    size_t workspace_size;
    float waves_count;
  };

  // Typedefs for cublas functions
  typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *);
  typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
  typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t *,
                                                       cublasComputeType_t,
                                                       cudaDataType_t);
  typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
  typedef cublasStatus_t (*cublasLtMatmulDescSetAttribute_t)(
      cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(
      cublasLtMatrixLayout_t *, cudaDataType_t, uint64_t, uint64_t, int64_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(
      cublasLtMatrixLayout_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(
      cublasLtMatmulPreference_t *);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(
      cublasLtMatmulPreference_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(
      cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t *,
      int *);
  typedef cublasStatus_t (*cublasLtMatmul_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *,
      const cublasLtMatrixLayout_t, const void *, const cublasLtMatrixLayout_t,
      const void *, const void *, const cublasLtMatrixLayout_t, void *,
      const cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *,
      size_t, cudaStream_t);
  typedef cublasStatus_t (*cublasLtMatmulAlgoGetIds_t)(
      cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t,
      cudaDataType_t, cudaDataType_t, cudaDataType_t, int, int *, int *);
  typedef cublasStatus_t (*cublasLtMatmulAlgoInit_t)(
      cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t,
      cudaDataType_t, cudaDataType_t, cudaDataType_t, int,
      cublasLtMatmulAlgo_t *);
  typedef cublasStatus_t (*cublasLtMatmulAlgoCapGetAttribute_t)(
      const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoCapAttributes_t, void *,
      size_t, size_t *);
  typedef cublasStatus_t (*cublasLtMatmulAlgoConfigSetAttribute_t)(
      cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t,
      const void *, size_t);
  typedef cublasStatus_t (*cublasLtMatmulAlgoCheck_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      const cublasLtMatmulAlgo_t *, cublasLtMatmulHeuristicResult_t *);
  typedef cublasStatus_t (*cublasLtMatmulAlgoConfigGetAttribute_t)(
      const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t,
      void *, size_t, size_t *);

  static constexpr const char *name = "libcublas.so";

  cublasLtCreate_t cublasLtCreate;
  cublasLtDestroy_t cublasLtDestroy;
  cublasLtMatmulDescCreate_t cublasLtMatmulDescCreate;
  cublasLtMatmulDescDestroy_t cublasLtMatmulDescDestroy;
  cublasLtMatmulDescSetAttribute_t cublasLtMatmulDescSetAttribute;
  cublasLtMatrixLayoutCreate_t cublasLtMatrixLayoutCreate;
  cublasLtMatrixLayoutDestroy_t cublasLtMatrixLayoutDestroy;
  cublasLtMatmulPreferenceCreate_t cublasLtMatmulPreferenceCreate;
  cublasLtMatmulPreferenceDestroy_t cublasLtMatmulPreferenceDestroy;
  cublasLtMatmulPreferenceSetAttribute_t cublasLtMatmulPreferenceSetAttribute;
  cublasLtMatmulAlgoGetHeuristic_t cublasLtMatmulAlgoGetHeuristic;
  cublasLtMatmul_t cublasLtMatmul;
  cublasLtMatmulAlgoGetIds_t cublasLtMatmulAlgoGetIds;
  cublasLtMatmulAlgoInit_t cublasLtMatmulAlgoInit;
  cublasLtMatmulAlgoCapGetAttribute_t cublasLtMatmulAlgoCapGetAttribute;
  cublasLtMatmulAlgoConfigSetAttribute_t cublasLtMatmulAlgoConfigSetAttribute;
  cublasLtMatmulAlgoCheck_t cublasLtMatmulAlgoCheck;
  cublasLtMatmulAlgoConfigGetAttribute_t cublasLtMatmulAlgoConfigGetAttribute;

  void *dylibHandle = nullptr;
  cublasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  cublasLtMatmulPreference_t preference = NULL;

  // Cached validated algorithms from the last get_algorithms_impl() call.
  // gemm_with_algo_impl() uses this to avoid re-querying heuristics.
  std::vector<ValidatedAlgo> cachedAlgos;
  int cachedM = 0, cachedN = 0, cachedK = 0;
  cudaDataType_t cachedDtype = static_cast<cudaDataType_t>(-1);

  void loadCublasDylib() {
    if (dylibHandle == nullptr) {
      // First reuse the existing handle
      dylibHandle = dlopen(name, RTLD_NOLOAD);
    }
    if (dylibHandle == nullptr) {
      // If not found, try to load it
      dylibHandle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (dylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Make sure it is in your "
                               "LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    cublasLtCreate = (cublasLtCreate_t)dlsym(dylibHandle, "cublasLtCreate");
    cublasLtDestroy = (cublasLtDestroy_t)dlsym(dylibHandle, "cublasLtDestroy");
    cublasLtMatmulDescCreate = (cublasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulDescCreate");
    cublasLtMatmulDescDestroy = (cublasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulDescDestroy");
    cublasLtMatmulDescSetAttribute = (cublasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "cublasLtMatmulDescSetAttribute");
    cublasLtMatrixLayoutCreate = (cublasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutCreate");
    cublasLtMatrixLayoutDestroy = (cublasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutDestroy");
    cublasLtMatmulPreferenceCreate = (cublasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceCreate");
    cublasLtMatmulPreferenceDestroy = (cublasLtMatmulPreferenceDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceDestroy");
    cublasLtMatmulPreferenceSetAttribute =
        (cublasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulPreferenceSetAttribute");
    cublasLtMatmulAlgoGetHeuristic = (cublasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoGetHeuristic");
    cublasLtMatmul = (cublasLtMatmul_t)dlsym(dylibHandle, "cublasLtMatmul");
    cublasLtMatmulAlgoGetIds = (cublasLtMatmulAlgoGetIds_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoGetIds");
    cublasLtMatmulAlgoInit = (cublasLtMatmulAlgoInit_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoInit");
    cublasLtMatmulAlgoCapGetAttribute =
        (cublasLtMatmulAlgoCapGetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulAlgoCapGetAttribute");
    cublasLtMatmulAlgoConfigSetAttribute =
        (cublasLtMatmulAlgoConfigSetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulAlgoConfigSetAttribute");
    cublasLtMatmulAlgoCheck = (cublasLtMatmulAlgoCheck_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoCheck");
    cublasLtMatmulAlgoConfigGetAttribute =
        (cublasLtMatmulAlgoConfigGetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulAlgoConfigGetAttribute");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadCublasDylib() {
    if (dylibHandle)
      dlclose(dylibHandle);
  }

  void successOrExit(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  // Simple wrapper around the cublasLtMatmul function
  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, cudaDataType_t dtype, float alpha, float beta) {
    cublasLtMatmulDesc_t matmulDesc = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // Select compute type. Use TF32 when inputs are FP32, otherwise default
    // FP32 accumulation.
    cublasComputeType_t computeType = (dtype == CUDA_R_32F)
                                          ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(cublasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    successOrExit(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic");
    }

    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, (void *)C, Cdesc,
                                 (void *)D, Ddesc, &heuristicResult.algo,
                                 (void *)workspace, workspaceSize, 0));
    if (Ddesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

  // Block-scaled matmul: D = (A * scale_A) @ (B * scale_B)
  //
  // Supports two modes via is_mxfp8 parameter:
  //   - MXFP8 (is_mxfp8=true):  FP8 E4M3 inputs, E8M0 scales (32-element
  //   groups)
  //   - NVFP4 (is_mxfp8=false): FP4 E2M1 inputs, FP8 E4M3 scales (16-element
  //   groups)
  //
  // Input layout requirements (row-major):
  //   - A: (M, K) in FP8/FP4 (FP4 is packed, 2 elements per byte)
  //   - B: (N, K) in FP8/FP4 (caller must transpose B before calling)
  //   - scale_A, scale_B: scale factors for block scaling
  //   - Output D: (M, N) in FP16
  //
  // Note: cuBLAS uses column-major layout. Similar to gemm_impl(), callers
  // should swap row-major operands and dimensions before invoking this helper.
  void block_scaled_matmul(int m, int n, int k, uint64_t A, uint64_t B,
                           uint64_t D_out, uint64_t scale_A, uint64_t scale_B,
                           bool is_mxfp8) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    // Use FP32 compute and accumulation
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Enable fast accumulation for MXFP8 only
    // "Flag for managing FP8 fast accumulation mode. When enabled, on some GPUs
    //  problem execution might be faster but at the cost of lower accuracy
    //  because intermediate results will not periodically be promoted to a
    //  higher precision. Currently this flag has an effect on the following
    //  GPUs: Ada, Hopper.""
    if (is_mxfp8) {
      int8_t fastAccum = 1;
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    // Set scale mode based on format
    // MXFP8: 32-element groups with E8M0 scales
    // NVFP4: 16-element groups with FP8 E4M3 scales
    cublasLtMatmulMatrixScale_t ab_scale_type =
        is_mxfp8 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ab_scale_type,
        sizeof(ab_scale_type)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &ab_scale_type,
        sizeof(ab_scale_type)));

    // Scale pointers follow the logical A/B operands that were passed in.
    void *scale_A_ptr = (void *)scale_A;
    void *scale_B_ptr = (void *)scale_B;
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_A_ptr,
        sizeof(scale_A_ptr)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_B_ptr,
        sizeof(scale_B_ptr)));

    // Create matrix layouts
    // MXFP8: CUDA_R_8F_E4M3, NVFP4: CUDA_R_4F_E2M1
    // With transa=T: A layout is (k, m), lda=k
    // With transb=N: B layout is (k, n), ldb=k
    cudaDataType_t dataType = is_mxfp8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, m));
    Ddesc = Cdesc;

    float alpha = 1.0f;
    float beta = 0.0f; // No bias

    // Query cuBLAS heuristics for the best algorithm
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
        &heuristicResult, &returnedResults);

    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
      throw std::runtime_error(
          "cublasLtMatmulAlgoGetHeuristic failed (status=" +
          std::to_string(status) +
          ", results=" + std::to_string(returnedResults) + ") for " +
          (is_mxfp8 ? "mxfp8" : "nvfp4"));
    }

    // Execute matmul with the selected algorithm using the already-swapped
    // row-major operands.
    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, (void *)D_out, Cdesc,
                                 (void *)D_out, Cdesc, &heuristicResult.algo,
                                 workspace, workspaceSize, 0));

    // Cleanup
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

  // Convert tile ID enum value to human-readable "MxN" string
  static std::string tileIdToString(uint32_t tileId) {
    static const char *names[] = {
        "UNDEFINED", "8x8",     "8x16",    "16x8",    "8x32",    "16x16",
        "32x8",      "8x64",    "16x32",   "32x16",   "64x8",    "32x32",
        "32x64",     "64x32",   "32x128",  "64x64",   "128x32",  "64x128",
        "128x64",    "64x256",  "128x128", "256x64",  "64x512",  "128x256",
        "256x128",   "512x64",  "64x96",   "96x64",   "96x128",  "128x160",
        "160x128",   "192x128", "128x192", "128x96",
    };
    if (tileId < sizeof(names) / sizeof(names[0]))
      return names[tileId];
    return "UNKNOWN(" + std::to_string(tileId) + ")";
  }

  // Build an expanded list of validated algorithms for the given problem.
  //
  // Two phases:
  //   1. Add each heuristic-recommended algo as-is (unmodified). These are the
  //      most reliable since the heuristic specifically chose them.
  //   2. Expand each heuristic algo across all its supported tile IDs,
  //      skipping tiles that match the original config. Each variant is
  //      validated with cublasLtMatmulAlgoCheck().
  //
  // The result is ordered: heuristic originals first, then expanded variants.
  std::vector<ValidatedAlgo>
  build_expanded_algos(cublasLtMatmulDesc_t matmulDesc,
                       cublasLtMatrixLayout_t Adesc,
                       cublasLtMatrixLayout_t Bdesc,
                       cublasLtMatrixLayout_t Cdesc,
                       cublasLtMatrixLayout_t Ddesc) {
    static constexpr int kMaxAlgos = 64;
    cublasLtMatmulHeuristicResult_t results[kMaxAlgos];
    int returnedResults = 0;
    successOrExit(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
        kMaxAlgos, results, &returnedResults));

    // Collect valid heuristic results
    std::vector<int> validIndices;
    for (int i = 0; i < returnedResults; ++i) {
      if (results[i].state == CUBLAS_STATUS_SUCCESS)
        validIndices.push_back(i);
    }

    std::vector<ValidatedAlgo> validated;

    // Phase 1: Add original heuristic algos unmodified (most reliable).
    for (int idx : validIndices) {
      const auto &hr = results[idx];
      if (hr.workspaceSize <= workspaceSize) {
        validated.push_back({hr.algo, hr.workspaceSize, hr.wavesCount});
      }
    }

    // Phase 2: Expand across supported tile IDs, skipping the original tile.
    for (int idx : validIndices) {
      cublasLtMatmulAlgo_t baseAlgo = results[idx].algo;

      // Read the original configured tile ID
      uint32_t originalTileId = 0;
      size_t sizeWritten = 0;
      cublasLtMatmulAlgoConfigGetAttribute(
          &baseAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &originalTileId,
          sizeof(originalTileId), &sizeWritten);

      // Get all supported tile IDs from capabilities
      sizeWritten = 0;
      std::vector<uint32_t> tileIds;
      if (cublasLtMatmulAlgoCapGetAttribute(
              &baseAlgo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0,
              &sizeWritten) == CUBLAS_STATUS_SUCCESS &&
          sizeWritten > 0) {
        tileIds.resize(sizeWritten / sizeof(uint32_t));
        cublasLtMatmulAlgoCapGetAttribute(&baseAlgo,
                                          CUBLASLT_ALGO_CAP_TILE_IDS,
                                          tileIds.data(), sizeWritten,
                                          &sizeWritten);
      }

      for (uint32_t tid : tileIds) {
        // Skip the tile already added in Phase 1
        if (tid == originalTileId)
          continue;

        cublasLtMatmulAlgo_t tileAlgo = baseAlgo;
        if (cublasLtMatmulAlgoConfigSetAttribute(
                &tileAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tid,
                sizeof(tid)) != CUBLAS_STATUS_SUCCESS)
          continue;

        cublasLtMatmulHeuristicResult_t checkResult = {};
        if (cublasLtMatmulAlgoCheck(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc,
                                    Ddesc, &tileAlgo,
                                    &checkResult) == CUBLAS_STATUS_SUCCESS &&
            checkResult.state == CUBLAS_STATUS_SUCCESS &&
            checkResult.workspaceSize <= workspaceSize) {
          validated.push_back({tileAlgo, checkResult.workspaceSize,
                               checkResult.wavesCount});
        }
      }
    }

    return validated;
  }

  // Query all available heuristic algorithms for the given problem.
  // Returns detailed information about each algorithm including its
  // configured tile, stages, and capability attributes.
  // Algorithms are expanded across valid tile configurations and verified
  // with cublasLtMatmulAlgoCheck() to ensure they will actually run.
  std::vector<AlgoHeuristicInfo>
  get_algorithms_impl(int m, int n, int k, cudaDataType_t dtype) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    cublasComputeType_t computeType = (dtype == CUDA_R_32F)
                                          ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(cublasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    auto validatedAlgos =
        build_expanded_algos(matmulDesc, Adesc, Bdesc, Cdesc, Ddesc);

    // Cache for later use by gemm_with_algo_impl
    cachedAlgos = validatedAlgos;
    cachedM = m;
    cachedN = n;
    cachedK = k;
    cachedDtype = dtype;

    std::vector<AlgoHeuristicInfo> algos;
    algos.reserve(validatedAlgos.size());

    for (size_t idx = 0; idx < validatedAlgos.size(); ++idx) {
      AlgoHeuristicInfo info;
      info.index = static_cast<int>(idx);
      info.workspace_size = validatedAlgos[idx].workspace_size;
      info.waves_count = validatedAlgos[idx].waves_count;

      const cublasLtMatmulAlgo_t &algo = validatedAlgos[idx].algo;
      uint32_t val = 0;
      size_t sizeWritten = 0;

      // Read configured tile ID
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS) {
        info.tile_id = val;
        info.tile_name = tileIdToString(val);
      } else {
        info.tile_id = 0;
        info.tile_name = "UNDEFINED";
      }

      // Read configured stages ID
      val = 0;
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS)
        info.stages_id = val;
      else
        info.stages_id = 0;

      // Read configured split-K
      val = 0;
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS)
        info.splitk_num = val;
      else
        info.splitk_num = 0;

      // Read configured reduction scheme
      val = 0;
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS)
        info.reduction_scheme = val;
      else
        info.reduction_scheme = 0;

      // Read configured CTA swizzling
      val = 0;
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS)
        info.cta_swizzling = val;
      else
        info.cta_swizzling = 0;

      // Read configured custom option
      val = 0;
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS)
        info.custom_option = val;
      else
        info.custom_option = 0;

      // Read capability: max custom option value
      val = 0;
      if (cublasLtMatmulAlgoCapGetAttribute(
              &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &val, sizeof(val),
              &sizeWritten) == CUBLAS_STATUS_SUCCESS)
        info.custom_option_max = val;
      else
        info.custom_option_max = 0;

      // Read capability: supported tile IDs
      sizeWritten = 0;
      if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
                                            NULL, 0,
                                            &sizeWritten) ==
              CUBLAS_STATUS_SUCCESS &&
          sizeWritten > 0) {
        std::vector<uint32_t> tileIds(sizeWritten / sizeof(uint32_t));
        if (cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileIds.data(), sizeWritten,
                &sizeWritten) == CUBLAS_STATUS_SUCCESS) {
          for (auto tid : tileIds)
            info.supported_tiles.push_back(tileIdToString(tid));
        }
      }

      // Read capability: supported stages IDs
      sizeWritten = 0;
      if (cublasLtMatmulAlgoCapGetAttribute(&algo,
                                            CUBLASLT_ALGO_CAP_STAGES_IDS, NULL,
                                            0, &sizeWritten) ==
              CUBLAS_STATUS_SUCCESS &&
          sizeWritten > 0) {
        info.supported_stages.resize(sizeWritten / sizeof(uint32_t));
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_STAGES_IDS,
            info.supported_stages.data(), sizeWritten, &sizeWritten);
      }

      algos.push_back(std::move(info));
    }

    if (Ddesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));

    return algos;
  }

  // Execute matmul using a specific algorithm previously cached by
  // get_algorithms_impl(). The caller must call get_algorithms_impl() first
  // to populate cachedAlgos; this method does NOT re-query heuristics.
  void gemm_with_algo_impl(int m, int n, int k, uint64_t A, uint64_t B,
                           uint64_t C, uint64_t D, cudaDataType_t dtype,
                           float alpha, float beta, int algo_index) {
    if (cachedAlgos.empty()) {
      throw std::runtime_error(
          "No cached algorithms. Call get_algorithms() before "
          "matmul_with_algo() or gemm_with_algo().");
    }
    if (m != cachedM || n != cachedN || k != cachedK || dtype != cachedDtype) {
      throw std::runtime_error(
          "Problem parameters (m=" + std::to_string(m) +
          ", n=" + std::to_string(n) +
          ", k=" + std::to_string(k) +
          ") do not match cached algorithms (m=" + std::to_string(cachedM) +
          ", n=" + std::to_string(cachedN) +
          ", k=" + std::to_string(cachedK) +
          "). Call get_algorithms() again for the new problem.");
    }
    if (algo_index < 0 ||
        algo_index >= static_cast<int>(cachedAlgos.size())) {
      throw std::runtime_error(
          "Algorithm index " + std::to_string(algo_index) +
          " out of range (only " +
          std::to_string(cachedAlgos.size()) +
          " valid algorithms available)");
    }

    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    cublasComputeType_t computeType = (dtype == CUDA_R_32F)
                                          ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(cublasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    successOrExit(cublasLtMatmul(
        ltHandle, matmulDesc, &alpha, (void *)A, Adesc, (void *)B, Bdesc,
        &beta, (void *)C, Cdesc, (void *)D, Ddesc,
        &cachedAlgos[algo_index].algo, (void *)workspace, workspaceSize, 0));

    if (Ddesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

public:
  CublasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadCublasDylib();
    cublasLtCreate(&ltHandle);

    successOrExit(cublasLtMatmulPreferenceCreate(&preference));
    successOrExit(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~CublasLtInstance() {
    if (preference)
      successOrExit(cublasLtMatmulPreferenceDestroy(preference));

    cublasLtDestroy(ltHandle);
    unloadCublasDylib();
  }

  // C = A * B
  // Matrix B needs to be transposed, while matrix A does not. The function
  // *will-not* transpose the matrices, so the caller is responsible for
  // ensuring that the matrices are in the correct format and have the correct
  // dimensions.
  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              cudaDataType_t dtype) {
    // CUDA is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    gemm_impl(n, m, k, B, A, 0, C, dtype, 1.0f, 0.0f);
  }

  void gemm(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
            cudaDataType_t dtype, float alpha, float beta) {
    gemm_impl(n, m, k, B, A, C, D, dtype, alpha, beta);
  }

  void block_scaled_matmul_mxfp8(int m, int n, int k, uint64_t A, uint64_t B,
                                 uint64_t D_out, uint64_t scale_A,
                                 uint64_t scale_B) {
    // Match gemm_impl()'s row-major handling by swapping operands and output
    // dimensions before calling into cuBLASLt's column-major API.
    block_scaled_matmul(n, m, k, B, A, D_out, scale_B, scale_A, true);
  }

  void block_scaled_matmul_nvfp4(int m, int n, int k, uint64_t A, uint64_t B,
                                 uint64_t D_out, uint64_t scale_A,
                                 uint64_t scale_B) {
    block_scaled_matmul(n, m, k, B, A, D_out, scale_B, scale_A, false);
  }

  // Get all available algorithms for the given problem size and dtype.
  // Returns a vector of AlgoHeuristicInfo with detailed attributes.
  std::vector<AlgoHeuristicInfo> get_algorithms(int m, int n, int k,
                                                cudaDataType_t dtype) {
    // CUDA is column-major, while triton is row-major, therefore we need to
    // reverse the order of the dimensions.
    return get_algorithms_impl(n, m, k, dtype);
  }

  // C = A * B using a specific algorithm from get_algorithms().
  // algo_index selects which algorithm to use (0 = best heuristic).
  //
  // NOTE: Not all algorithms returned by get_algorithms() are guaranteed to
  // run successfully. Heuristic-recommended algorithms (low indices) are the
  // most reliable, but even they may fail at runtime due to alignment or
  // hardware constraints. Tile-expanded variants (higher indices) pass
  // cublasLtMatmulAlgoCheck() but cuBLAS may still reject them at execution
  // time with CUBLAS_STATUS_NOT_SUPPORTED. Callers should be prepared to
  // catch RuntimeError and fall back to a different algorithm.
  void matmul_with_algo(int m, int n, int k, uint64_t A, uint64_t B,
                        uint64_t C, cudaDataType_t dtype, int algo_index) {
    gemm_with_algo_impl(n, m, k, B, A, 0, C, dtype, 1.0f, 0.0f, algo_index);
  }

  // D = alpha * A * B + beta * C using a specific algorithm.
  // algo_index selects which algorithm to use (0 = best heuristic).
  //
  // NOTE: Same caveats as matmul_with_algo() — not all algorithms are
  // guaranteed to succeed at runtime. See matmul_with_algo() for details.
  void gemm_with_algo(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                      uint64_t D, cudaDataType_t dtype, float alpha, float beta,
                      int algo_index) {
    gemm_with_algo_impl(n, m, k, B, A, C, D, dtype, alpha, beta, algo_index);
  }
};

#endif // TRITON_CUBLAS_INSTANCE_H
