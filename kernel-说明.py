from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

        x_ptr: 输入张量的指针。
        y_ptr: 输出（量化后）张量的指针。
        s_ptr: 存储每个块的缩放因子的张量指针。
        BLOCK_SIZE: 每个线程块处理的数据大小，是一个编译时常量。

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32) #从内存中加载数据到局部变量 x 中，并转换为浮点32位类型以确保精度。
    s = tl.max(tl.abs(x)) / 448. #计算当前块的最大绝对值，并除以448得到缩放因子 s。
    y = x / s #将原始数据 x 除以缩放因子 s 得到量化前的值 y。
    y = y.to(y_ptr.dtype.element_ty) #将结果 y 转换为目标数据类型（这里是 float8_e4m3fn），
    tl.store(y_ptr + offs, y) #然后存储回输出张量 y_ptr 的相应位置。
    tl.store(s_ptr + pid, s) #将计算出的缩放因子存储在 s_ptr 中对应的位置。


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        x: 输入张量，需要是连续存储的，并且最后一个维度的大小必须能被 block_size 整除。
        block_size: 每个块的大小，默认值为128。
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous' #检查输入张量是否满足要求：必须是连续存储的，
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    #最后一个维度的大小必须能够被 block_size 整除。
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)#创建一个与输入张量形状相同的空张量 y，但数据类型为 torch.float8_e4m3fn
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    #创建一个用于存储缩放因子的张量 s，用于存储每个块的缩放因子，除最后一维形状和x一致，最后一维为x.size(-1) // block_size。
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )#定义一个网格配置函数 grid，它决定了如何划分任务给不同的线程块。
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size) #执行量化操作；
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]
# 这是一个包含多个配置的列表，用于自动调优（autotuning）Triton 内核。
# 每个配置定义了不同的块大小（BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K）以及内核执行的阶段数（num_stages）和 warp 数（num_warps）。
# 目的是找到最优的参数组合以提高性能。

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])#自动调优装饰器，根据输入张量的 N 和 K 维度选择最佳配置。
@triton.jit#标记这是一个 JIT 编译的 Triton 内核函数。
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

        •a_ptr, b_ptr, c_ptr: 输入矩阵 A、B 和输出矩阵 C 的指针。
        •a_s_ptr, b_s_ptr: 输入矩阵 A 和 B 的缩放因子指针。
        •M, N, K: 矩阵维度。
        •BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: 块大小。

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)#计算当前线程块的索引 (pid_m, pid_n)。
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)#初始化偏移量 (offs_m, offs_n, offs_k) 用于访问数据。
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k
    ##加载输入矩阵 A 和 B 的子块，并应用相应的缩放因子。

    #执行矩阵乘法并累加结果到 accumulator。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    #将最终结果存储回输出矩阵 C。
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        a, b: 输入矩阵 A 和 B，必须是连续存储的。
        a_s, b_s: 输入矩阵 A 和 B 的缩放因子，也必须是连续存储的。

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    #检查所有输入张量是否连续存储。
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())#创建一个空的输出张量 c。
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N'])) #定义网格配置函数 grid，它决定了如何划分任务给不同的线程块。
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)#调用 Triton 内核函数 fp8_gemm_kernel 来执行矩阵乘法操作。
    return c
    """
    关键点总结
    •FP8 精度：代码假设支持 FP8 数据类型，这在标准 PyTorch 中可能不可直接支持，需要特定硬件或库的支持。
    •自动调优：通过尝试不同的块大小和执行阶段数来优化性能。
    •Triton 内核：利用 Triton 提供的高度并行化能力来加速矩阵乘法操作。
    •缩放因子：考虑到浮点精度损失，使用缩放因子来调整数值范围，从而保持计算精度。
    这段代码展示了如何结合 Triton 的高效并行计算能力和 PyTorch 的灵活性来实现高效的矩阵乘法操作，特别是在处理低精度数据类型时。
    """
