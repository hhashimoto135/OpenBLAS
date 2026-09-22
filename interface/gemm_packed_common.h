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

/* Shared helpers for the CBLAS entry points of the packed GEMM extension
 * (interface/gemm_pack.c and interface/gemm_compute.c). */

#ifndef GEMM_PACKED_COMMON_H
#define GEMM_PACKED_COMMON_H

#include <stdio.h>
#include "common.h"
#include "common_sbfallback.h"

#if defined(COMPLEX) || defined(XDOUBLE) || defined(HFLOAT16) || defined(BGEMM)
#error "the packed GEMM interface supports SGEMM, DGEMM, and SBGEMM only"
#endif

#ifndef CBLAS
#error "the packed GEMM extension provides a CBLAS interface only"
#endif

#define GEMM_PACKED_CONCAT_(a, b) a##b
#define GEMM_PACKED_CONCAT(a, b) GEMM_PACKED_CONCAT_(a, b)

#if defined(BFLOAT16)
#define GEMM_PACKED_SIZE    sbgemm_packed_size
#define GEMM_PACKED_PACK    sbgemm_packed_pack
#define GEMM_PACKED_COMPUTE sbgemm_packed_compute
#define GEMM_PACKED_PREFIX  "SBGEMM"
#define GEMM_PACKED_TAG     GEMM_PACKED_TAG_SB
#elif defined(DOUBLE)
#define GEMM_PACKED_SIZE    dgemm_packed_size
#define GEMM_PACKED_PACK    dgemm_packed_pack
#define GEMM_PACKED_COMPUTE dgemm_packed_compute
#define GEMM_PACKED_PREFIX  "DGEMM"
#define GEMM_PACKED_TAG     GEMM_PACKED_TAG_D
#else
#define GEMM_PACKED_SIZE    sgemm_packed_size
#define GEMM_PACKED_PACK    sgemm_packed_pack
#define GEMM_PACKED_COMPUTE sgemm_packed_compute
#define GEMM_PACKED_PREFIX  "SGEMM"
#define GEMM_PACKED_TAG     GEMM_PACKED_TAG_S
#endif

/* Maps a CBLAS transpose value to the 0/1 code of the driver. Real types treat
 * the conjugate variants as their plain counterparts, as interface/gemm.c
 * does. Returns -1 for anything else. */
static inline int gemm_packed_trans_code(blasint trans) {
  switch (trans) {
    case CblasNoTrans:
    case CblasConjNoTrans:
      return 0;
    case CblasTrans:
    case CblasConjTrans:
      return 1;
    default:
      return -1;
  }
}

/* Under DYNAMIC_ARCH the blocking parameters live in the gotoblas table,
 * which a constructor fills before main() with GCC and clang. MSVC has no
 * constructor, and there the table is filled by the first blas_memory_alloc();
 * cblas_?gemm_pack_get_size never allocates, so it fills the table itself if
 * it is the first call into the library. */
static inline void gemm_packed_ensure_initialized(void) {
#ifdef DYNAMIC_ARCH
  if (gotoblas == NULL) gotoblas_dynamic_init();
#endif
}

/* Returns 0 when the entry point may go on: the bfloat16 kernels of the core
 * in use can run (sbgemm_kernels_unavailable() first replaces them with
 * Cooperlake's when that is what it takes), or the single precision fallback
 * of common_sbfallback.h stands in for them. Non-zero only when neither
 * holds; the caller then returns OPENBLAS_GEMM_STATUS_NO_KERNEL. Always 0 for
 * SGEMM and DGEMM. Every bfloat16 entry point, cblas_sbgemm_pack_get_size
 * included, calls this before it reads an SBGEMM blocking parameter. */
static inline int gemm_packed_kernels_ready(void) {
#if defined(BFLOAT16) && defined(ARCH_X86_64)
  if (sbgemm_kernels_unavailable()) {
#if defined(SBGEMM_FLOAT_FALLBACK)
    return 0;
#else
    return -1;
#endif
  }
#endif
  return 0;
}

#endif /* GEMM_PACKED_COMMON_H */
