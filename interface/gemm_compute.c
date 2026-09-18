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
 * cblas_?gemm_compute: the last third of the MKL-style packed GEMM API.
 *
 *   cblas_?gemm_compute(order, transa, transb, m, n, k,
 *                       a, lda, b, ldb, beta, c, ldc)
 *
 * transa and transb accept the CBLAS_TRANSPOSE values or CblasPacked. A
 * CblasPacked operand points at a buffer filled by cblas_?gemm_pack with the
 * same identifier, order, m/n (its own extent), and k; its leading dimension
 * argument is ignored. The routine computes
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * where alpha is the product of the alphas recorded in the packed operand(s),
 * or 1 when neither operand is packed.
 *
 * The scratch buffer for the operand that is not packed comes from
 * blas_memory_alloc, exactly as for a regular GEMM, so concurrent calls from
 * several threads are safe and may share one read-only packed buffer.
 */

#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "common_sbfallback.h"
#include "gemm_packed_common.h"

#define ERROR_NAME GEMM_PACKED_PREFIX "_COMPUTE "

void CNAME(enum CBLAS_ORDER order, blasint transa, blasint transb, blasint m, blasint n, blasint k,
           IFLOAT *a, blasint lda, IFLOAT *b, blasint ldb, FLOAT beta, FLOAT *c, blasint ldc) {
  blas_arg_t args;
  blasint info = 0;
  int a_packed, b_packed, ta, tb;
  int internal_ta = 0, internal_tb = 0, internal_a_packed = 0, internal_b_packed = 0;
  int bad_ld8 = 0, bad_ld10 = 0;
  int row_major;
  BLASLONG nrowa, nrowb;
  FLOAT alpha = ONE;
  XFLOAT *buffer, *sa, *sb;
  int status;

  PRINT_DEBUG_CNAME;

  a_packed = (transa == CblasPacked);
  b_packed = (transb == CblasPacked);
  ta = a_packed ? 0 : gemm_packed_trans_code(transa);
  tb = b_packed ? 0 : gemm_packed_trans_code(transb);

  args.alpha = (void *)&alpha;
  args.beta  = (void *)&beta;
  args.c     = (void *)c;
  args.ldc   = ldc;
  args.k     = k;

  row_major = (order == CblasRowMajor);

  if (order == CblasColMajor) {
    args.m = m;
    args.n = n;
    args.a = (void *)a;
    args.b = (void *)b;
    args.lda = lda;
    args.ldb = ldb;
    internal_ta = ta;
    internal_tb = tb;
    internal_a_packed = a_packed;
    internal_b_packed = b_packed;
  } else if (row_major) {
    /* C^T = op(B)^T op(A)^T: the operands change roles, and so do m and n. */
    args.m = n;
    args.n = m;
    args.a = (void *)b;
    args.b = (void *)a;
    args.lda = ldb;
    args.ldb = lda;
    internal_ta = tb;
    internal_tb = ta;
    internal_a_packed = b_packed;
    internal_b_packed = a_packed;
  } else {
    args.m = m;
    args.n = n;
    args.a = (void *)a;
    args.b = (void *)b;
    args.lda = lda;
    args.ldb = ldb;
  }

  /* Leading dimensions are only meaningful for operands that are not packed.
   * The checks run on the internal operands and are reported against the
   * user's argument positions: internal lda is the user's ldb (10) in
   * row-major order and lda (8) otherwise. */
  nrowa = internal_ta ? (BLASLONG)k : args.m;
  nrowb = internal_tb ? args.n : (BLASLONG)k;
  if (!internal_a_packed && args.lda < nrowa) {
    if (row_major) bad_ld10 = 1; else bad_ld8 = 1;
  }
  if (!internal_b_packed && args.ldb < nrowb) {
    if (row_major) bad_ld8 = 1; else bad_ld10 = 1;
  }

  /* Later assignments win, so the lowest offending argument is reported. A
   * NULL operand is only an error when the product reads it. */
  if (args.ldc < args.m) info = 13;
  if (bad_ld10) info = 10;
  if (b == NULL && (b_packed || (k > 0 && n > 0))) info = 9;
  if (bad_ld8) info = 8;
  if (a == NULL && (a_packed || (k > 0 && m > 0))) info = 7;
  if (k < 0) info = 6;
  if (n < 0) info = 5;
  if (m < 0) info = 4;
  if (tb < 0) info = 3;
  if (ta < 0) info = 2;
  if (order != CblasColMajor && order != CblasRowMajor) info = 1;

  if (info) {
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

  if ((args.m == 0) || (args.n == 0)) return;

  if (gemm_packed_kernels_ready() != 0) return;

  buffer = (XFLOAT *)blas_memory_alloc(0);

#if defined(ARCH_LOONGARCH64) && !defined(NO_AFFINITY)
  sa = (XFLOAT *)((BLASLONG)buffer + (WhereAmI() & 0xf) * GEMM_OFFSET_A);
#else
  sa = (XFLOAT *)((BLASLONG)buffer + GEMM_OFFSET_A);
#endif
#ifdef SBGEMM_FLOAT_FALLBACK
  /* The fallback runs the SGEMM kernels on float panels, so the scratch is
   * divided where a regular SGEMM divides it, not where SBGEMM would. */
  if (sbgemm_float_fallback()) {
    sb = (XFLOAT *)(((BLASLONG)sa + (((BLASLONG)SGEMM_P * (BLASLONG)SGEMM_Q * (BLASLONG)sizeof(float)
                                      + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
  } else
#endif
  sb = (XFLOAT *)(((BLASLONG)sa + ((GEMM_P * GEMM_Q * COMPSIZE * SIZE + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);

  status = GEMM_PACKED_COMPUTE(&args, internal_ta, internal_tb, internal_a_packed, internal_b_packed,
                               sa, sb, GEMM_PACKED_TAG);

  blas_memory_free(buffer);

  if (status != 0) {
    /* Bit 0: the internal A buffer was rejected, bit 1: the internal B buffer.
     * Map back to the user's operand positions a (7) and b (9) and report the
     * lower one, as the argument checks above do. */
    int user_a_bad = row_major ? (status & 2) : (status & 1);
    info = user_a_bad ? 7 : 9;
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
  }
}
