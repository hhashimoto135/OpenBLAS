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
 * Decides, once per process, whether the bfloat16 GEMM kernels of the core in
 * use can run, and what takes their place when they cannot.
 *
 * Only the Sapphire Rapids kernels have a condition beyond the instruction
 * set. They use AMX tile instructions, which the OS has to enable (XCR0 bits
 * 17 and 18) and which Linux grants per process through
 * arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA); a kernel older than
 * 5.16, or a container that filters the syscall, refuses. Upstream asks on the
 * first SBGEMM call and, when refused, prints a message and returns without
 * computing anything.
 *
 * The request stays lazy here, so a process that never touches bfloat16 gets
 * no AMX permission and none of its side effects (the kernel sizes signal
 * frames for the permitted features), but the product is never given up:
 *
 *   - DYNAMIC_ARCH: the bfloat16 GEMM entries of the table in use are
 *     replaced by Cooperlake's, whose kernels need AVX512-BF16 only
 *     (gotoblas_sbgemm_use_cooperlake() in dynamic.c). Nothing else in the
 *     table changes, so every other routine, including one running in another
 *     thread at that moment, is unaffected. The SBGEMM fields are read only
 *     by the bfloat16 GEMM entry points, and each of those calls
 *     sbgemm_kernels_unavailable() before it reads any of them, so no caller
 *     can see the fields half way through the rewrite.
 *   - Otherwise (a static SAPPHIRERAPIDS build, or a DYNAMIC_LIST build with
 *     no Cooperlake table of its own) the caller is told that the kernels are
 *     unavailable, and the single precision fallback of common_sbfallback.h
 *     computes the product with the SGEMM kernels, which need AVX512 only.
 *
 * The answer is cached, so a pack and the compute that follows cannot disagree
 * about the layout of a packed buffer.
 */

#include "common.h"

extern void openblas_warning(int verbose, const char *msg);

#if defined(ARCH_X86_64) && defined(BUILD_BFLOAT16)

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

#define SBGEMM_AMX_UNRESOLVED  0
#define SBGEMM_AMX_READY       1
#define SBGEMM_AMX_UNAVAILABLE 2

static volatile int sbgemm_amx_state = SBGEMM_AMX_UNRESOLVED;
static volatile BLASULONG sbgemm_amx_lock = 0;

#if defined(DYNAMIC_ARCH)

/* Both live in dynamic.c: the CPUID / XCR0 probe that get_coretype() uses,
 * and the rewrite of the bfloat16 GEMM entries. */
extern int support_amx_bf16(void);
extern int gotoblas_sbgemm_use_cooperlake(void);
#define sbgemm_amx_supported support_amx_bf16

#elif defined(SAPPHIRERAPIDS)

/* The static build has no dynamic.c. The probe repeats what
 * support_amx_bf16() there asks: AMX-TILE and AMX-BF16 in CPUID.7.0:EDX, and
 * the tile state enabled by the OS in XCR0. */
static int sbgemm_amx_supported(void) {
  int eax, ebx, ecx, edx;

  cpuid(1, &eax, &ebx, &ecx, &edx);
  if ((ecx & (1 << 27)) == 0) return 0; /* OSXSAVE, needed for xgetbv */
  cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
  if ((edx & (1 << 24)) == 0 || (edx & (1 << 22)) == 0) return 0;
  __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0) : "cc");
  return (eax & 0x00060000) == 0x00060000;
}

#endif

#if defined(DYNAMIC_ARCH) || defined(SAPPHIRERAPIDS)
/* Non-zero when this process may execute AMX tile instructions. */
static int sbgemm_amx_usable(void) {
  if (!sbgemm_amx_supported()) return 0;
#if defined(__linux__)
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) return 0;
#endif
  return 1;
}
#endif

static int sbgemm_amx_resolve(void) {
#if defined(DYNAMIC_ARCH)
  if (gotoblas == NULL) gotoblas_dynamic_init();
  if (!gotoblas->need_amxtile_permission) return SBGEMM_AMX_READY;
  if (sbgemm_amx_usable()) return SBGEMM_AMX_READY;

  if (gotoblas_sbgemm_use_cooperlake() == 0) {
    openblas_warning(1, "OpenBLAS : AMX is not available to this process. Using the Cooperlake bfloat16 GEMM kernels instead.\n");
    return SBGEMM_AMX_READY;
  }
  openblas_warning(0, "OpenBLAS : AMX is not available to this process, and this build has no other bfloat16 kernel.\n");
  return SBGEMM_AMX_UNAVAILABLE;
#elif defined(SAPPHIRERAPIDS)
  if (sbgemm_amx_usable()) return SBGEMM_AMX_READY;
  openblas_warning(0, "OpenBLAS : AMX is not available to this process, so the Sapphire Rapids bfloat16 kernels cannot run.\n");
  return SBGEMM_AMX_UNAVAILABLE;
#else
  return SBGEMM_AMX_READY;
#endif
}

/*
 * Returns 0 when the bfloat16 GEMM kernels of the core in use may run, after
 * replacing them with Cooperlake's if that is what it takes, and non-zero when
 * they cannot run in this process at all. The callers then take the single
 * precision fallback of common_sbfallback.h where it is compiled in, and
 * otherwise return without touching their output.
 */
int sbgemm_kernels_unavailable(void) {
  int state = sbgemm_amx_state;

  if (state == SBGEMM_AMX_UNRESOLVED) {
    blas_lock(&sbgemm_amx_lock);
    state = sbgemm_amx_state;
    if (state == SBGEMM_AMX_UNRESOLVED) {
      state = sbgemm_amx_resolve();
      /* The table must be complete before anyone can see the state. */
      MB;
      sbgemm_amx_state = state;
    }
    blas_unlock(&sbgemm_amx_lock);
  }
  return state == SBGEMM_AMX_UNAVAILABLE;
}

#else /* not x86-64, or no bfloat16 */

int sbgemm_kernels_unavailable(void) {
  return 0;
}

#endif
