import pytest
import torch
from triton._internal_testing import is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.parametrize("m, n, k", [(16, 16, 16), (32, 16, 16), (16, 32, 16), (16, 16, 32)])
@pytest.mark.parametrize("dtype_str", ["float8_e4m3fn", "float8_e4m3fnuz", "float16"])
def test_blaslt(m, n, k, dtype_str, device):
    dtype = getattr(torch, dtype_str)

    if is_cuda():
        from triton._C.libtriton import nvidia as vendor
        if dtype_str == "float8_e4m3fnuz":
            pytest.skip("float8_e4m3fnuz is not supported on CUDA")
        if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("fp8 is only supported on CUDA with cc >= 90")
        c_dtype = dtype
        make_handle = lambda workspace: vendor.cublas.CublasLt(workspace)
    elif is_hip():
        from triton._C.libtriton import amd as vendor
        if dtype_str == "float8_e4m3fnuz" and not is_hip_cdna3():
            pytest.skip("float8_e4m3fnuz is only supported on HIP CDNA3")
        if dtype_str == "float8_e4m3fn" and not is_hip_cdna4():
            pytest.skip("float8_e4m3fn is only supported on HIP CDNA4")
        c_dtype = torch.float16 if dtype_str in ("float8_e4m3fnuz", "float8_e4m3fn") else dtype
        make_handle = lambda workspace: vendor.hipblas.HipblasLt(workspace)
    else:
        pytest.skip("test_blaslt is only supported on CUDA or HIP")

    torch.manual_seed(123)
    workspace_size = 32 * 1024 * 1024

    def limited_rand(elements, shape):
        total_elems = torch.prod(torch.tensor(shape)).item()
        indices = torch.randint(0, len(elements), (total_elems, ), device=device)
        return elements[indices].view(shape)

    elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
    a = limited_rand(elements, (m, k)).to(dtype)
    b = limited_rand(elements, (k, n)).to(dtype)

    c = torch.zeros((m, n), dtype=c_dtype, device=device)

    b = b.T.contiguous()

    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)
    handle = make_handle(workspace)

    handle.matmul(a, b, c)

    ref = torch.matmul(a.to(torch.float16), b.to(torch.float16).T)

    assert torch.allclose(c.to(torch.float16), ref, atol=2.0)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (256, 256, 512),
        (256, 512, 512),
        (512, 256, 512),
        (512, 512, 512),
        (1024, 1024, 1024),
    ],
)
def test_block_scaled_matmul_mxfp8(m, n, k, device):
    """Test block-scaled matmul with MXFP8 format (FP8 E4M3 inputs, E8M0 scales)."""
    if not is_cuda():
        pytest.skip("block_scaled_matmul is only supported on CUDA")
    if not supports_block_scaling():
        pytest.skip("block_scaled_matmul requires compute capability 10.0 (Blackwell)")

    from triton._C.libtriton import nvidia

    torch.manual_seed(42)

    # Constants for MXFP8
    VEC_SIZE = 32  # 32-element groups for E8M0 scales

    # Create workspace and cuBLAS handle
    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    # Generate random FP8 inputs
    a_fp32 = torch.randn(m, k, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)

    # Convert to FP8 E4M3
    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e4m3fn)

    # Generate scales in the expected 4D layout, then reshape to 5D and flatten
    # Scale shape: [M // 128, K // VEC_SIZE // 4, 32, 16]
    a_scale_shape = [m // 128, k // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [n // 128, k // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale_raw = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale_raw = torch.rand(b_scale_shape, device=device) + epsilon

    # Convert to MXScaleTensor (E8M0 format)
    a_scale_mx = MXScaleTensor(a_scale_raw)
    b_scale_mx = MXScaleTensor(b_scale_raw)
    a_scale = a_scale_mx.data
    b_scale = b_scale_mx.data

    # Reshape to 5D for TMA and flatten for cuBLAS
    a_scale_5d = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale_5d = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_cublas = a_scale_5d.contiguous().flatten()
    b_scale_cublas = b_scale_5d.contiguous().flatten()

    # Prepare output tensor
    output = torch.empty((m, n), dtype=torch.float16, device=device)

    # Call cuBLAS block-scaled matmul
    handle.block_scaled_matmul_mxfp8(a, b, output, a_scale_cublas, b_scale_cublas)

    # Compute reference using PyTorch
    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_ref = a_scale_mx.to(torch.float32)
    b_scale_ref = b_scale_mx.to(torch.float32)
    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:m, :k]
    b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:k, :n]

    ref = torch.matmul(a.to(torch.float32) * a_scale_ref, b.to(torch.float32).T * b_scale_ref)

    torch.testing.assert_close(output.to(torch.float32), ref, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("m, n, k", [(256, 256, 512), (512, 512, 512), (1024, 1024, 1024)])
def test_block_scaled_matmul_nvfp4(m, n, k, device):
    """Test block-scaled matmul with NVFP4 format (packed FP4 inputs, FP8 E4M3 scales)."""
    if not is_cuda():
        pytest.skip("block_scaled_matmul is only supported on CUDA")
    if not supports_block_scaling():
        pytest.skip("block_scaled_matmul requires compute capability 10.0 (Blackwell)")

    from triton._C.libtriton import nvidia

    torch.manual_seed(42)

    # Constants for NVFP4
    VEC_SIZE = 16  # 16-element groups for FP8 E4M3 scales

    # Create workspace and cuBLAS handle
    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    # Generate random MXFP4 tensors
    a_ref = MXFP4Tensor(size=(m, k), device=device).random()
    b_ref = MXFP4Tensor(size=(n, k), device=device).random()

    # Pack two FP4 elements per byte along K dimension
    a = a_ref.to_packed_tensor(dim=1)  # (M, K//2) in uint8
    b = b_ref.to_packed_tensor(dim=1)  # (N, K//2) in uint8

    # Generate scales in the expected 4D layout
    # Scale shape: [M // 128, K // VEC_SIZE // 4, 32, 16]
    a_scale_shape = [m // 128, k // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [n // 128, k // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale_raw = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale_raw = torch.rand(b_scale_shape, device=device) + epsilon

    # For NVFP4, scales are FP8 E4M3
    a_scale = a_scale_raw.to(torch.float8_e4m3fn)
    b_scale = b_scale_raw.to(torch.float8_e4m3fn)

    # Flatten for cuBLAS (use original 4D layout, not 5D reshaped)
    a_scale_cublas = a_scale.contiguous().flatten()
    b_scale_cublas = b_scale.contiguous().flatten()

    # Prepare output tensor
    output = torch.empty((m, n), dtype=torch.float16, device=device)

    # Call cuBLAS block-scaled matmul
    handle.block_scaled_matmul_nvfp4(a, b, output, a_scale_cublas, b_scale_cublas)

    # Compute reference using PyTorch
    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_ref = a_scale.to(torch.float32)
    b_scale_ref = b_scale.to(torch.float32)
    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:m, :k]
    b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:k, :n]

    ref = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref.to(torch.float32).T * b_scale_ref)

    torch.testing.assert_close(output.to(torch.float32), ref, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("m, n, k", [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)])
@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
def test_get_algorithms(m, n, k, dtype_str, device):
    """Test that get_algorithms returns a non-empty list of algorithm details."""
    if not is_cuda():
        pytest.skip("get_algorithms is only supported on CUDA")

    from triton._C.libtriton import nvidia

    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    algos = handle.get_algorithms(m, n, k, dtype_str)

    assert len(algos) > 0, "Expected at least one algorithm"

    # Verify structure of each algorithm info dict
    for algo in algos:
        assert "index" in algo
        assert "workspace_size" in algo
        assert "waves_count" in algo
        assert "tile_id" in algo
        assert "tile_name" in algo
        assert "stages_id" in algo
        assert "splitk_num" in algo
        assert "reduction_scheme" in algo
        assert "cta_swizzling" in algo
        assert "custom_option" in algo
        assert "custom_option_max" in algo
        assert "supported_tiles" in algo
        assert "supported_stages" in algo

    # Verify that tile expansion produces algos with diverse tile configs
    tile_names = set(algo['tile_name'] for algo in algos)
    print(f"\n  Problem: {m}x{n}x{k} ({dtype_str})")
    print(f"  Found {len(algos)} algorithms with {len(tile_names)} distinct tiles: {sorted(tile_names)}")
    for algo in algos[:5]:  # Print first 5
        print(f"    [{algo['index']}] tile={algo['tile_name']}, "
              f"stages={algo['stages_id']}, "
              f"waves={algo['waves_count']:.2f}, "
              f"workspace={algo['workspace_size']}")
    # With tile expansion we expect more than one distinct tile (for non-trivial sizes)
    if m >= 256 and n >= 256:
        assert len(tile_names) > 1, (
            f"Expected diverse tile configs from expansion, got only {tile_names}")


@pytest.mark.parametrize("m, n, k", [(256, 256, 256), (512, 512, 512)])
@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
def test_matmul_with_algo(m, n, k, dtype_str, device):
    """Test matmul_with_algo produces correct results for multiple algorithm indices."""
    if not is_cuda():
        pytest.skip("matmul_with_algo is only supported on CUDA")

    from triton._C.libtriton import nvidia

    dtype = getattr(torch, dtype_str)
    torch.manual_seed(123)

    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    # First query available algorithms
    algos = handle.get_algorithms(m, n, k, dtype_str)
    assert len(algos) > 0

    # Use small integer values to avoid accumulation errors
    def limited_rand(shape):
        elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
        total_elems = torch.prod(torch.tensor(shape)).item()
        indices = torch.randint(0, len(elements), (total_elems,), device=device)
        return elements[indices].view(shape)

    a = limited_rand((m, k)).to(dtype)
    b = limited_rand((k, n)).to(dtype)

    # B must be transposed (N, K) layout, same as matmul
    b_t = b.T.contiguous()

    # Reference: standard matmul
    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(dtype)

    # Test algorithms. Not all algorithms are guaranteed to succeed at runtime
    # — even those passing cublasLtMatmulAlgoCheck() may fail with
    # CUBLAS_STATUS_NOT_SUPPORTED due to alignment or hardware constraints.
    # We require at least one algorithm to work.
    num_to_test = min(5, len(algos))
    num_passed = 0
    for i in range(num_to_test):
        c = torch.zeros((m, n), dtype=dtype, device=device)
        try:
            handle.matmul_with_algo(a, b_t, c, i)
        except RuntimeError as e:
            print(f"\n  algo[{i}] tile={algos[i]['tile_name']}, "
                  f"stages={algos[i]['stages_id']} => SKIPPED ({e})")
            continue

        torch.testing.assert_close(c.to(torch.float32), ref.to(torch.float32), atol=2.0, rtol=1e-2)
        num_passed += 1
        print(f"\n  algo[{i}] tile={algos[i]['tile_name']}, "
              f"stages={algos[i]['stages_id']} => PASS")

    assert num_passed > 0, (
        f"All {num_to_test} tested algorithms failed at runtime")


@pytest.mark.parametrize("dtype_str", ["float16"])
def test_matmul_with_algo_requires_get_algorithms(dtype_str, device):
    """Test that matmul_with_algo raises an error if get_algorithms was not called first."""
    if not is_cuda():
        pytest.skip("matmul_with_algo is only supported on CUDA")

    from triton._C.libtriton import nvidia

    dtype = getattr(torch, dtype_str)
    m, n, k = 256, 256, 256

    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    a = torch.randn((m, k), dtype=dtype, device=device)
    b_t = torch.randn((n, k), dtype=dtype, device=device)
    c = torch.zeros((m, n), dtype=dtype, device=device)

    # Should fail: no prior get_algorithms() call
    with pytest.raises(RuntimeError, match="No cached algorithms"):
        handle.matmul_with_algo(a, b_t, c, 0)


@pytest.mark.parametrize("dtype_str", ["float16"])
def test_matmul_with_algo_stale_cache(dtype_str, device):
    """Test that matmul_with_algo raises an error if problem params changed since get_algorithms."""
    if not is_cuda():
        pytest.skip("matmul_with_algo is only supported on CUDA")

    from triton._C.libtriton import nvidia

    dtype = getattr(torch, dtype_str)

    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    # Cache algos for 256x256x256
    algos = handle.get_algorithms(256, 256, 256, dtype_str)
    assert len(algos) > 0

    # Now try to use cached algos with a different problem size
    a = torch.randn((512, 512), dtype=dtype, device=device)
    b_t = torch.randn((512, 512), dtype=dtype, device=device)
    c = torch.zeros((512, 512), dtype=dtype, device=device)

    with pytest.raises(RuntimeError, match="do not match cached algorithms"):
        handle.matmul_with_algo(a, b_t, c, 0)
