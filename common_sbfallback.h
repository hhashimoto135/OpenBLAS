/*********************************************************************/
/* Copyright 2026 The OpenBLAS Project                               */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS SOFTWARE IS PROVIDED BY THE OPENBLAS PROJECT ``AS IS''    */
/*    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      */
/*    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND      */
/*    FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT   */
/*    SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY   */
/*    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR           */
/*    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,          */
/*    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,      */
/*    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED     */
/*    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT    */
/*    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)         */
/*    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF    */
/*    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                     */
/*********************************************************************/

/*
 * Single precision fallback for SBGEMM on x86-64 cores that have no bfloat16
 * GEMM kernel, or whose bfloat16 kernel cannot run in this process.
 *
 * kernel/Makefile.L3 leaves SBGEMMKERNEL at ../generic/gemmkernel_2x2.c with
 * ../generic/gemm_ncopy_2.c and ../generic/gemm_tcopy_2.c for every core that
 * does not set it. On x86-64 only KERNEL.COOPERLAKE and KERNEL.SAPPHIRERAPIDS
 * do, so on everything else SBGEMM runs a scalar 2x2 C kernel while SGEMM runs
 * a tuned one. That kernel widens every element to float with SBF16TOS_K
 * before multiplying, so expanding both operands the same way up front and
 * running the SGEMM kernels computes the same product at SGEMM speed. The
 * price is a float copy of each operand, twice the bytes of the bfloat16
 * original, and a summation order that differs from both the 2x2 kernel and
 * the Cooperlake kernel, so results agree to rounding rather than bit for bit.
 *
 * The widening preserves every normal value; it is SBF16TOS_K that decides the
 * rest, flushing denormals to a zero of the same sign and making signalling
 * NaNs quiet. sbgemm_widen() below repeats those two rules so that the
 * fallback and the 2x2 kernel see identical operands.
 *
 * The condition is the core selected at run time, not the instruction set of
 * the CPU: OPENBLAS_CORETYPE can pin a core without a bfloat16 kernel on a CPU
 * that supports AVX512-BF16, and that core is what decides which kernel runs.
 * The Sapphire Rapids kernel has one more condition, AMX being usable by the
 * process; sbgemm_kernels_unavailable() (driver/others/sbgemm_amx.c) settles
 * it, replaces the bfloat16 GEMM entries of the table with Cooperlake's where
 * it can, and otherwise hands the product to this fallback.
 *
 * Only x86-64 is covered. Other architectures either ship a bfloat16 kernel
 * (POWER10, Neoverse N2, RISC-V ZVL) or are outside what this fork targets.
 */

#ifndef COMMON_SBFALLBACK_H
#define COMMON_SBFALLBACK_H

#if defined(ARCH_X86_64)
/* Non-zero when the bfloat16 GEMM kernels of the core in use cannot run in
 * this process, which on x86-64 means Sapphire Rapids without usable AMX.
 * Under DYNAMIC_ARCH the bfloat16 GEMM entries of the table are first replaced
 * by Cooperlake's when the build has that table, so the answer is non-zero
 * only where no bfloat16 kernel can take over and the fallback below is the
 * last resort. Every bfloat16 GEMM entry point asks before it reads an SBGEMM
 * blocking parameter or kernel pointer, and the answer is cached for the life
 * of the process. */
int sbgemm_kernels_unavailable(void);
#endif

#if defined(BFLOAT16) && !defined(BGEMM) && defined(ARCH_X86_64) && defined(BUILD_SINGLE)
#define SBGEMM_FLOAT_FALLBACK 1
#endif

#ifdef SBGEMM_FLOAT_FALLBACK

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef DYNAMIC_ARCH
extern char *gotoblas_corename(void);
#endif

extern void openblas_warning(int verbose, const char *msg);

#define SBGEMM_FALLBACK_NO_MEMORY "sbgemm: cannot allocate the float expansion\n"

/* Non-zero when SBGEMM must be computed with the SGEMM kernels. */
static inline int sbgemm_float_fallback(void) {
#if defined(DYNAMIC_ARCH)
  const char *core;
  /* Resolve AMX first. A Sapphire Rapids table whose AMX is unusable keeps
   * its name but runs Cooperlake's bfloat16 kernels afterwards, so the name
   * still says that a bfloat16 kernel exists. */
  if (sbgemm_kernels_unavailable()) return 1;
  core = gotoblas_corename();
  return strcmp(core, "Cooperlake") != 0 && strcmp(core, "SapphireRapids") != 0;
#elif defined(SAPPHIRERAPIDS)
  return sbgemm_kernels_unavailable();
#elif defined(COOPERLAKE)
  return 0;
#else
  return 1;
#endif
}

/*
 * Widens `n` contiguous bfloat16 values to float.
 *
 * This is SBF16TOS_K written without the per element switch, so that the
 * compiler can vectorize it: kernel/x86_64/bf16to.c is scalar on every x86-64
 * core and costs more than the GEMM it would feed. The three cases it treats
 * specially are kept exactly - a denormal (biased exponent 0) flushes to a
 * zero of the same sign, a NaN gets its quiet bit forced, and everything else
 * is the bfloat16 pattern in the upper half of the float.
 */
static inline void sbgemm_widen(const bfloat16 *src, float *dest, BLASLONG n) {
  BLASLONG i;

  for (i = 0; i < n; i++) {
    uint32_t bits = (uint32_t)src[i];
    uint32_t exponent = bits & 0x7f80u;
    /* All ones or all zeros, so that the two corrections are plain masks. */
    uint32_t denormal = (uint32_t)0 - (uint32_t)(exponent == 0);
    uint32_t nan = (uint32_t)0 - (uint32_t)(exponent == 0x7f80u && (bits & 0x007fu) != 0);

    bits = (bits & (0x8000u | ~denormal)) | (nan & 0x0040u);
    bits <<= 16;
    memcpy(dest + i, &bits, sizeof(bits));
  }
}

/*
 * Expands `rows` x `cols` of the column-major bfloat16 matrix `src`, whose
 * leading dimension is `ld`, into a freshly allocated compact column-major
 * float matrix with leading dimension `rows`. Returns NULL when the size does
 * not fit or the allocation fails; free the result with free().
 *
 * An empty matrix still returns a one element allocation, so that NULL keeps
 * meaning failure at the call sites.
 */
static inline float *sbgemm_expand_to_float(bfloat16 *src, BLASLONG rows, BLASLONG cols, BLASLONG ld) {
  size_t count;
  float *dest;
  BLASLONG j;

  if (rows < 0 || cols < 0) return NULL;
  if (cols != 0 && (size_t)rows > (SIZE_MAX / sizeof(float)) / (size_t)cols) return NULL;

  count = (size_t)rows * (size_t)cols;
  dest = (float *)malloc((count != 0 ? count : 1) * sizeof(float));
  if (dest == NULL) return NULL;
  /* Nothing to read, and `src` may legitimately be NULL for an empty
   * operand. */
  if (count == 0) return dest;

  if (ld == rows) {
    /* Contiguous source: one pass with no column boundary to interrupt it. */
    sbgemm_widen(src, dest, (BLASLONG)count);
  } else {
    for (j = 0; j < cols; j++) {
      sbgemm_widen(src + j * ld, dest + j * rows, rows);
    }
  }
  return dest;
}

/* Leading dimension to report for an expanded operand. BLAS rejects a leading
 * dimension below 1 even when the matrix is empty. */
static inline BLASLONG sbgemm_expanded_ld(BLASLONG rows) {
  return rows > 0 ? rows : 1;
}

#endif /* SBGEMM_FLOAT_FALLBACK */

#endif /* COMMON_SBFALLBACK_H */
