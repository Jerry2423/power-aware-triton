"""cuBLAS Lt based F.linear (output = xA^T + b) with algorithm selection.

Usage:
    fn = CublasLtLinear()
    algos = fn.get_algorithms(m, n, k, "float16")   # must call first
    output = fn(x, weight, bias)                     # uses algo_index=0 by default
"""

import torch

# torch dtype → cuBLAS Lt dtype string
_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
}


class CublasLtLinear:
    """Implements ``torch.nn.functional.linear`` via cuBLAS Lt.

    Internally dispatches to:
    * ``gemm_with_algo``   – when *bias* is provided  (D = αA·Bᵀ + βC)
    * ``matmul_with_algo`` – when *bias* is ``None``   (C = A·Bᵀ)

    Both require a prior ``get_algorithms`` call so that cuBLAS Lt knows
    which algorithm to use.
    """

    @staticmethod
    def _default_workspace_size() -> int:
        """Return the cuBLAS-Lt workspace size that matches PyTorch's heuristic.

        * Hopper  (sm_90),  Blackwell (sm_100 / sm_120) → 32 MiB
        * All other architectures                        → ~8.25 MiB
        """
        major = torch.cuda.get_device_capability()[0]
        # mirror PyTorch: 4096*8*1024 for Hopper/Blackwell, else 4096*1024*2 + 16*1024*8
        if major in (9, 10, 12):
            return 4096 * 8 * 1024            # 32 MiB
        return 4096 * 1024 * 2 + 16 * 1024 * 8  # ~8.25 MiB

    def __init__(self, workspace_size: int | None = None):
        from triton._C.libtriton import nvidia

        if workspace_size is None:
            workspace_size = self._default_workspace_size()
        self.workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device="cuda"
        )
        self.handle = nvidia.cublas.CublasLt(self.workspace)
        self._algos = None
        self._algo_key = None

    # ------------------------------------------------------------------
    # Algorithm query
    # ------------------------------------------------------------------
    def get_algorithms(self, m: int, n: int, k: int, dtype):
        """Query cuBLAS Lt for available algorithms.

        **Must** be called before :meth:`linear` / :meth:`__call__`.

        Parameters
        ----------
        m, n, k : int
            Problem dimensions  (output = [m, n],  A = [m, k],  B = [n, k]).
        dtype : torch.dtype | str
            Element type, e.g. ``torch.float16`` or ``"float16"``.

        Returns
        -------
        list[dict]
            Algorithm descriptors sorted by cuBLAS Lt heuristic (index 0 is
            the recommended pick).
        """
        if isinstance(dtype, torch.dtype):
            dtype_str = _DTYPE_TO_STR[dtype]
        else:
            dtype_str = dtype
        self._algo_key = (m, n, k, dtype_str)
        self._algos = self.handle.get_algorithms(m, n, k, dtype_str)
        return self._algos

    @property
    def algorithms(self):
        """Return the most recently cached algorithm list (or ``None``)."""
        return self._algos

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(self, x, weight, bias=None, algo_index=0):
        return self.linear(x, weight, bias, algo_index)

    def linear(self, x, weight, bias=None, algo_index: int = 0):
        """Compute ``output = x @ weight.T [+ bias]``.

        Parameters
        ----------
        x : Tensor[m, k]
            Input activations.
        weight : Tensor[n, k]
            Weight matrix (same layout as ``nn.Linear.weight``).
        bias : Tensor[n] | None
            Optional bias.  Will be broadcast to ``[m, n]`` and cast to
            ``float16`` (cuBLAS Lt constraint).
        algo_index : int
            Index into the list returned by :meth:`get_algorithms`.
            ``0`` selects the best (first) algorithm.

        Returns
        -------
        Tensor[m, n]
        """
        if self._algos is None:
            raise RuntimeError(
                "No algorithms cached. "
                "Call get_algorithms(m, n, k, dtype) before linear()."
            )

        m, k = x.shape
        n = weight.shape[0]

        # cuBLAS Lt requires contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()

        if bias is not None:
            # gemm_with_algo: D = alpha * A @ B^T + beta * C
            # Constraints: C must be float16, C.shape == D.shape
            bias_2d = bias.unsqueeze(0).expand(m, n).contiguous()
            if bias_2d.dtype != torch.float16:
                bias_2d = bias_2d.to(torch.float16)
            output = torch.empty((m, n), dtype=x.dtype, device=x.device)
            self.handle.gemm_with_algo(
                x, weight, bias_2d, output, 1.0, 1.0, algo_index
            )
        else:
            # matmul_with_algo: C = A @ B^T
            output = torch.zeros((m, n), dtype=x.dtype, device=x.device)
            self.handle.matmul_with_algo(x, weight, output, algo_index)

        return output
