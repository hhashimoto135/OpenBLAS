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
#elif defined(DOUBLE)
#define GEMM_PACKED_SIZE    dgemm_packed_size
#define GEMM_PACKED_PACK    dgemm_packed_pack
#define GEMM_PACKED_COMPUTE dgemm_packed_compute
#define GEMM_PACKED_PREFIX  "DGEMM"
#else
#define GEMM_PACKED_SIZE    sgemm_packed_size
#define GEMM_PACKED_PACK    sgemm_packed_pack
#define GEMM_PACKED_COMPUTE sgemm_packed_compute
#define GEMM_PACKED_PREFIX  "SGEMM"
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

/* AMX tile data must be enabled per process on Linux before the Sapphire
 * Rapids SBGEMM copy and compute kernels may touch tile registers. This is the
 * same request interface/gemm.c makes. Returns 0 when the kernels may run. */
#if defined(__linux__) && defined(__x86_64__) && defined(BFLOAT16)
#include <unistd.h>
#include <sys/syscall.h>
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
/* The permission is granted per process (Documentation/arch/x86/xstate.rst),
 * so a successful request is remembered. The flag is only ever set to 1 and
 * read with relaxed atomics, so concurrent first calls at worst repeat the
 * request, which the kernel answers with success. */
static int gemm_packed_amxtile_permission = 0;

static inline int gemm_packed_request_amxtile(void) {
  long status;
  if (__atomic_load_n(&gemm_packed_amxtile_permission, __ATOMIC_RELAXED)) return 0;
  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (status != 0) {
    fprintf(stderr, "XTILEDATA permission not granted in your device(Linux, "
                    "Intel Sapphier Rapids), skip sbgemm calculation\n");
    return -1;
  }
  __atomic_store_n(&gemm_packed_amxtile_permission, 1, __ATOMIC_RELAXED);
  return 0;
}
#endif

static inline int gemm_packed_kernels_ready(void) {
#if defined(__linux__) && defined(__x86_64__) && defined(BFLOAT16)
#if defined(DYNAMIC_ARCH)
  if (gotoblas->need_amxtile_permission && gemm_packed_request_amxtile() == -1) return -1;
#elif defined(SAPPHIRERAPIDS)
  if (gemm_packed_request_amxtile() == -1) return -1;
#endif
#endif
  return 0;
}

#endif /* GEMM_PACKED_COMMON_H */
