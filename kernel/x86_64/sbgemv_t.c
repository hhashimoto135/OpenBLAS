/***************************************************************************
Copyright (c) 2014, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/


#include "common.h"

extern void openblas_warning(int verbose, const char *msg);

#if defined (COOPERLAKE) || defined (SAPPHIRERAPIDS)
#include "sbgemv_t_microk_cooperlake.c"
#endif

#define ALIGN64_ALLOC(alloc_size, TYPE, ptr_align, ptr)   \
    ptr = (TYPE *) malloc(sizeof(TYPE)*alloc_size + 63); \
    ptr_align = (ptr == NULL) ? NULL : ((int)(((uintptr_t)ptr & (uintptr_t)0x3F))!=0) ? (TYPE *)((char *)ptr + (64 - (int)((uintptr_t)ptr & (uintptr_t)0x3F))) : ptr

#define ALIGN64_FREE(ptr) \
    free(ptr)

#ifndef HAVE_SBGEMV_T_ACCL_KERNEL
/*
 * Scalar fallback for cores without a bfloat16 GEMV kernel. It reads the
 * operands in place, widening each element as SBF16TOS_K does (a denormal
 * becomes a zero of the same sign, a NaN gets its quiet bit), so it needs no
 * copy of the matrix and cannot fail for want of memory. The products are
 * summed in the same order as before, one row of the transposed matrix at a
 * time over j. m and n are the swapped extents, as in CNAME below.
 */
static inline float sbgemv_widen(bfloat16 v)
{
    uint32_t bits = (uint32_t)v;
    uint32_t exponent = bits & 0x7f80u;
    uint32_t denormal = (uint32_t)0 - (uint32_t)(exponent == 0);
    uint32_t nan = (uint32_t)0 - (uint32_t)(exponent == 0x7f80u && (bits & 0x007fu) != 0);
    float out;

    bits = (bits & (0x8000u | ~denormal)) | (nan & 0x0040u);
    bits <<= 16;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void sbgemv_kernel_t_strided(BLASLONG m, BLASLONG n, float alpha, bfloat16 *a, BLASLONG lda,
                                    bfloat16 *x, BLASLONG incx, float beta, float *y, BLASLONG incy)
{
    for (BLASLONG i=0; i<m; i++) {
        BLASLONG offset_lda = lda * i;
        float accum = 0.0;
        for (BLASLONG j=0; j<n; j++) {
            accum += sbgemv_widen(a[offset_lda + j]) * sbgemv_widen(x[j*incx]);
        }
        if (beta == ZERO) {
            y[i*incy] = alpha * accum;
        } else {
            y[i*incy] = alpha * accum + beta * y[i*incy];
        }
    }
}
#endif

#ifdef HAVE_SBGEMV_T_ACCL_KERNEL
static void bf16_compress_vector(BLASLONG n, bfloat16 * src, bfloat16 * target, BLASLONG inc)
{
    for(BLASLONG i=0; i<n; i++) {
        target[i] = src[i*inc];
    }
}

static void fp32_compress_vector(BLASLONG n, float * src, float * target, BLASLONG inc)
{
    for(BLASLONG i=0; i<n; i++) {
        target[i] = src[i*inc];
    }
}

static void fp32_expand_vector(BLASLONG n, float * src, float * target, BLASLONG inc)
{
    for(BLASLONG i=0; i<n; i++) {
        target[i*inc] = src[i];
    }
}
#endif

int CNAME(BLASLONG m, BLASLONG n, float alpha, bfloat16 *a, BLASLONG lda, bfloat16 *x, BLASLONG incx, float beta, float * y, BLASLONG incy)
{
    if ( m < 1 || n < 1) return(0);

    // Switch m and n
    BLASLONG t = m;
    m = n;
    n = t;

#ifndef HAVE_SBGEMV_T_ACCL_KERNEL
    sbgemv_kernel_t_strided(m, n, alpha, a, lda, x, incx, beta, y, incy);
    return(0);
#else
    bfloat16 * xbuffer_align = x;
    float    * ybuffer_align = y;

    bfloat16 * xbuffer = NULL;
    float    * ybuffer = NULL;

    /* The accelerated kernel reads contiguous vectors. A strided x or y is
     * compacted into a temporary; when that cannot be allocated, y is left
     * untouched and the thread's failure flag is set (openblas_alloc_failed). */
    if (incx != 1) {
        ALIGN64_ALLOC(n, bfloat16, xbuffer_align, xbuffer);
        if (xbuffer == NULL) {
            openblas_warning(0, "sbgemv: cannot allocate a vector copy, the call is skipped\n");
            blas_memory_note_failure();
            return(1);
        }
        bf16_compress_vector(n, x, xbuffer_align, incx);
    }

    if (incy != 1) {
        ALIGN64_ALLOC(m, float, ybuffer_align, ybuffer);
        if (ybuffer == NULL) {
            if (incx != 1) {
                ALIGN64_FREE(xbuffer);
            }
            openblas_warning(0, "sbgemv: cannot allocate a vector copy, the call is skipped\n");
            blas_memory_note_failure();
            return(1);
        }
        if (beta != ZERO) {
            fp32_compress_vector(m, y, ybuffer_align, incy);
        }
    }

    sbgemv_kernel_t(m, n, alpha, a, lda, xbuffer_align, beta, ybuffer_align);

    if (incy != 1) {
        fp32_expand_vector(m, ybuffer_align, y, incy);
        ALIGN64_FREE(ybuffer);
    }

    if (incx != 1) {
        ALIGN64_FREE(xbuffer);
    }

    return(0);
#endif
}
