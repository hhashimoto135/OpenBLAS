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
 * cblas_?gemm_pack_get_size and cblas_?gemm_pack: the first two thirds of the
 * MKL-style packed GEMM API. This file is compiled once per type with CBLAS
 * defined and CNAME set to cblas_sgemm_pack, cblas_dgemm_pack, or
 * cblas_sbgemm_pack; the _get_size entry point derives its name from CNAME.
 *
 *   size   = cblas_?gemm_pack_get_size(identifier, m, n, k)
 *   status = cblas_?gemm_pack(order, identifier, trans, m, n, k, alpha, src, ld, dest)
 *
 * The packed buffer records alpha; the product computed later by
 * cblas_?gemm_compute is alpha * op(A) * op(B) + beta * C, with alpha taken
 * from the packed operand(s). When both operands are packed the two recorded
 * alphas are multiplied.
 *
 * cblas_?gemm_pack returns 0 when dest was packed. Otherwise the value says
 * why not: the number of the rejected argument, or one of the negative
 * OPENBLAS_GEMM_STATUS_* codes of cblas.h. A rejected argument and
 * OPENBLAS_GEMM_STATUS_TOO_LARGE leave dest untouched; the other failures
 * happen after writing began and leave no valid header behind, so that a
 * later cblas_?gemm_compute rejects the buffer. MKL declares the routine void;
 * callers written for MKL may keep ignoring the result.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "gemm_packed_common.h"

#define ERROR_NAME_PACK     GEMM_PACKED_PREFIX "_PACK "
#define ERROR_NAME_GET_SIZE GEMM_PACKED_PREFIX "_PACK_GET_SIZE "

#define GEMM_PACK_GET_SIZE_NAME GEMM_PACKED_CONCAT(CNAME, _get_size)

size_t GEMM_PACK_GET_SIZE_NAME(enum CBLAS_IDENTIFIER identifier, blasint m, blasint n, blasint k) {
  blasint info = 0;
  BLASLONG extent;
  size_t bytes;

  PRINT_DEBUG_CNAME;

  if (k < 0) info = 4;
  if (n < 0) info = 3;
  if (m < 0) info = 2;
  if (identifier != CblasAMatrix && identifier != CblasBMatrix) info = 1;

  if (info) {
    BLASFUNC(xerbla)(ERROR_NAME_GET_SIZE, &info, sizeof(ERROR_NAME_GET_SIZE));
    return 0;
  }

  gemm_packed_ensure_initialized();
  /* The size follows the blocking of the kernels that will pack, so the
   * bfloat16 kernel question is settled first. A size is still returned when
   * the kernels cannot run; cblas_?gemm_pack reports that. */
  (void)gemm_packed_kernels_ready();

  extent = (identifier == CblasAMatrix) ? (BLASLONG)m : (BLASLONG)n;
  bytes = GEMM_PACKED_SIZE(extent, (BLASLONG)k);
  if (bytes == 0) {
    /* The packed size does not fit in memory. There is no single offending
     * argument, so the report uses parameter 0, as other size errors do. */
    info = 0;
    BLASFUNC(xerbla)(ERROR_NAME_GET_SIZE, &info, sizeof(ERROR_NAME_GET_SIZE));
  }
  return bytes;
}

int CNAME(enum CBLAS_ORDER order, enum CBLAS_IDENTIFIER identifier, enum CBLAS_TRANSPOSE trans,
          blasint m, blasint n, blasint k, FLOAT alpha, IFLOAT *src, blasint ld, IFLOAT *dest) {
  blasint info = 0;
  int internal_identifier = -1;
  int internal_trans, status;
  BLASLONG internal_m = 0, internal_n = 0, rows;

  PRINT_DEBUG_CNAME;

  internal_trans = gemm_packed_trans_code((blasint)trans);

  /* The driver works in column-major orientation. A row-major matrix is the
   * transpose of the column-major matrix with the same storage, so the two
   * operands change roles and m and n swap, exactly as in interface/gemm.c. */
  if (order == CblasColMajor) {
    internal_m = m;
    internal_n = n;
    if (identifier == CblasAMatrix) internal_identifier = GEMM_PACKED_IDENTIFIER_A;
    if (identifier == CblasBMatrix) internal_identifier = GEMM_PACKED_IDENTIFIER_B;
  } else if (order == CblasRowMajor) {
    internal_m = n;
    internal_n = m;
    if (identifier == CblasAMatrix) internal_identifier = GEMM_PACKED_IDENTIFIER_B;
    if (identifier == CblasBMatrix) internal_identifier = GEMM_PACKED_IDENTIFIER_A;
  }

  /* Rows of the stored source matrix, which bound the leading dimension. */
  rows = 0;
  if (internal_identifier == GEMM_PACKED_IDENTIFIER_A) rows = internal_trans ? (BLASLONG)k : internal_m;
  if (internal_identifier == GEMM_PACKED_IDENTIFIER_B) rows = internal_trans ? internal_n : (BLASLONG)k;

  /* Later assignments win, so the lowest offending argument is reported. */
  if (dest == NULL) info = 10;
  if (ld < rows) info = 9;
  if (src == NULL && k > 0 && (internal_identifier == GEMM_PACKED_IDENTIFIER_A ? internal_m : internal_n) > 0) info = 8;
  if (k < 0) info = 6;
  if (n < 0) info = 5;
  if (m < 0) info = 4;
  if (internal_trans < 0) info = 3;
  if (identifier != CblasAMatrix && identifier != CblasBMatrix) info = 2;
  if (order != CblasColMajor && order != CblasRowMajor) info = 1;

  if (info) {
    BLASFUNC(xerbla)(ERROR_NAME_PACK, &info, sizeof(ERROR_NAME_PACK));
    return (int)info;
  }

  gemm_packed_ensure_initialized();

  if (gemm_packed_kernels_ready() != 0) {
    /* The bfloat16 kernels cannot run in this process and nothing stands in
     * for them (see interface/gemm.c, which gives up the same way). Leave no
     * valid header behind, so that a later cblas_?gemm_compute rejects the
     * buffer instead of reading stale panels. */
    memset(dest, 0, GEMM_PACKED_HEADER_BYTES);
    return OPENBLAS_GEMM_STATUS_NO_KERNEL;
  }

  status = GEMM_PACKED_PACK(internal_identifier, internal_trans, internal_m, internal_n, (BLASLONG)k,
                            alpha, src, (BLASLONG)ld, (void *)dest);
  switch (status) {
    case 0:
      return 0;
    case 4:
      /* The float expansion of the single precision SBGEMM fallback could not
       * be allocated. The driver has said so and zeroed the header; this is
       * not an argument error, so xerbla is not involved. */
      return OPENBLAS_GEMM_STATUS_NO_MEMORY;
    case 2:
      /* The packed size does not fit, so cblas_?gemm_pack_get_size had
       * returned 0 for these dimensions. */
      info = 0;
      BLASFUNC(xerbla)(ERROR_NAME_PACK, &info, sizeof(ERROR_NAME_PACK));
      return OPENBLAS_GEMM_STATUS_TOO_LARGE;
    default:
      /* 3: the panels written disagree with the size computed in closed form;
       * 1 cannot happen after the checks above. Both are internal errors. */
      info = 0;
      BLASFUNC(xerbla)(ERROR_NAME_PACK, &info, sizeof(ERROR_NAME_PACK));
      return OPENBLAS_GEMM_STATUS_INTERNAL;
  }
}
