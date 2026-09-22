/*****************************************************************************
Copyright (c) 2026, The OpenBLAS Project
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
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

#include "utest/openblas_utest.h"
#include "common.h"

#ifdef BUILD_BFLOAT16
#ifndef NO_CBLAS

#define PACK_SENTINEL 0x5A
#define PACK_SENTINEL_BYTES 16

/* See test_sgemm_pack.c: malloc with a sentinel tail, or, with guard_pages set
 * (Linux only), mmap placed so that the buffer ends at a PROT_NONE page. */
static int guard_pages = FALSE;

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>

struct guard_map {
    unsigned char *ptr;
    void *base;
    size_t len;
};
static struct guard_map guard_maps[2];

static unsigned char *guard_alloc(size_t size)
{
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t pages = (size + page - 1) / page;
    void *base = mmap(NULL, (pages + 1) * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int slot = guard_maps[0].ptr == NULL ? 0 : 1;
    if (base == MAP_FAILED) return NULL;
    mprotect((char *)base + pages * page, page, PROT_NONE);
    guard_maps[slot].base = base;
    guard_maps[slot].len = (pages + 1) * page;
    guard_maps[slot].ptr = (unsigned char *)base + pages * page - size;
    return guard_maps[slot].ptr;
}

static void guard_free(unsigned char *ptr)
{
    int slot;
    for (slot = 0; slot < 2; slot++) {
        if (guard_maps[slot].ptr == ptr) {
            munmap(guard_maps[slot].base, guard_maps[slot].len);
            guard_maps[slot].ptr = NULL;
            return;
        }
    }
}
#endif

static unsigned char *alloc_packed(size_t size)
{
    unsigned char *buffer;
    if (size == 0) return NULL;
#if defined(__linux__)
    if (guard_pages) return guard_alloc(size);
#endif
    buffer = (unsigned char *)malloc(size + PACK_SENTINEL_BYTES);
    if (buffer != NULL) memset(buffer + size, PACK_SENTINEL, PACK_SENTINEL_BYTES);
    return buffer;
}

static void free_packed(unsigned char *buffer)
{
    if (buffer == NULL) return;
#if defined(__linux__)
    if (guard_pages) {
        guard_free(buffer);
        return;
    }
#endif
    free(buffer);
}

static int sentinel_intact(const unsigned char *buffer, size_t size)
{
    size_t i;
    if (guard_pages) return TRUE;
    for (i = 0; i < PACK_SENTINEL_BYTES; i++)
        if (buffer[size + i] != PACK_SENTINEL) return FALSE;
    return TRUE;
}

/* Random bfloat16 values in [0, 1). The conversion truncates the float bits
 * instead of calling cblas_sbstobf16, whose threaded path does not return for
 * n > 100000 when OPENBLAS_NUM_THREADS=1 (an unrelated upstream issue). */
static bfloat16 to_bf16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bfloat16)(bits >> 16);
}

static bfloat16 *random_bf16(size_t count)
{
    float *values = (float *)malloc((count + 1) * sizeof(float));
    bfloat16 *result = (bfloat16 *)malloc((count + 1) * sizeof(bfloat16));
    size_t i;
    srand_generate(values, (blasint)count);
    for (i = 0; i < count; i++) result[i] = to_bf16(values[i]);
    free(values);
    return result;
}

/**
 * Compute the reference with cblas_sbgemm, pack the requested operands with
 * cblas_sbgemm_pack, run cblas_sbgemm_compute, and return the largest
 * absolute difference divided by (1 + largest reference magnitude).
 *
 * When only one operand is packed it records alpha. When both are packed, A
 * records alpha and B records 3, so that the product of the two recorded
 * values, which cblas_sbgemm_compute applies, differs from either of them.
 */
static float check_sbgemm_pack(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa,
                               enum CBLAS_TRANSPOSE transb, blasint m, blasint n, blasint k,
                               float alpha, float beta, int pack_a, int pack_b, blasint pad)
{
    int row_major = (order == CblasRowMajor);
    blasint a_rows = (transa == CblasNoTrans) ? m : k;
    blasint a_cols = (transa == CblasNoTrans) ? k : m;
    blasint b_rows = (transb == CblasNoTrans) ? k : n;
    blasint b_cols = (transb == CblasNoTrans) ? n : k;
    blasint lda = (row_major ? a_cols : a_rows) + pad;
    blasint ldb = (row_major ? b_cols : b_rows) + pad;
    blasint ldc = (row_major ? n : m) + pad;
    size_t a_size = (size_t)(row_major ? a_rows : a_cols) * (size_t)lda;
    size_t b_size = (size_t)(row_major ? b_rows : b_cols) * (size_t)ldb;
    size_t c_size = (size_t)(row_major ? m : n) * (size_t)ldc;
    bfloat16 *a = random_bf16(a_size);
    bfloat16 *b = random_bf16(b_size);
    float *c_ref = (float *)malloc((c_size + 1) * sizeof(float));
    float *c_test = (float *)malloc((c_size + 1) * sizeof(float));
    unsigned char *packed_a = NULL, *packed_b = NULL;
    size_t size_a = 0, size_b = 0;
    float alpha_a = pack_a ? alpha : 1.0f;
    float alpha_b = pack_b ? (pack_a ? 3.0f : alpha) : 1.0f;
    float ref_alpha = alpha_a * alpha_b;
    float max_diff = 0.0f, max_ref = 0.0f;
    size_t i;

    srand_generate(c_ref, (blasint)c_size);
    for (i = 0; i < c_size; i++) c_test[i] = c_ref[i];

    cblas_sbgemm(order, transa, transb, m, n, k, ref_alpha, a, lda, b, ldb, beta, c_ref, ldc);

    if (pack_a) {
        size_a = cblas_sbgemm_pack_get_size(CblasAMatrix, m, n, k);
        packed_a = alloc_packed(size_a);
        cblas_sbgemm_pack(order, CblasAMatrix, transa, m, n, k, alpha_a, a, lda, (bfloat16 *)packed_a);
    }
    if (pack_b) {
        size_b = cblas_sbgemm_pack_get_size(CblasBMatrix, m, n, k);
        packed_b = alloc_packed(size_b);
        cblas_sbgemm_pack(order, CblasBMatrix, transb, m, n, k, alpha_b, b, ldb, (bfloat16 *)packed_b);
    }

    cblas_sbgemm_compute(order, pack_a ? (blasint)CblasPacked : (blasint)transa,
                         pack_b ? (blasint)CblasPacked : (blasint)transb, m, n, k,
                         pack_a ? (bfloat16 *)packed_a : a, lda,
                         pack_b ? (bfloat16 *)packed_b : b, ldb, beta, c_test, ldc);

    for (i = 0; i < c_size; i++) {
        float diff = fabsf(c_test[i] - c_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (fabsf(c_ref[i]) > max_ref) max_ref = fabsf(c_ref[i]);
    }

    if ((pack_a && !sentinel_intact(packed_a, size_a)) ||
        (pack_b && !sentinel_intact(packed_b, size_b)))
        max_diff = 1.0e30f;

    free(a);
    free(b);
    free(c_ref);
    free(c_test);
    free_packed(packed_a);
    free_packed(packed_b);

    return max_diff / (1.0f + max_ref);
}

CTEST(sbgemm_pack, get_size_is_positive)
{
    ASSERT_TRUE(cblas_sbgemm_pack_get_size(CblasAMatrix, 7, 9, 11) > 0);
    ASSERT_TRUE(cblas_sbgemm_pack_get_size(CblasBMatrix, 7, 9, 11) > 0);
}

CTEST(sbgemm_pack, rowmajor_pack_b_notrans_notrans)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 50, 70, 30,
                                  1.5f, 2.0f, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

CTEST(sbgemm_pack, rowmajor_pack_a_trans_notrans)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasTrans, CblasNoTrans, 50, 70, 30,
                                  0.5f, 0.0f, TRUE, FALSE, 3);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

CTEST(sbgemm_pack, rowmajor_pack_both_trans_trans)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasTrans, CblasTrans, 33, 65, 17,
                                  2.0f, 1.0f, TRUE, TRUE, 1);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

CTEST(sbgemm_pack, colmajor_pack_b_notrans_trans)
{
    float err = check_sbgemm_pack(CblasColMajor, CblasNoTrans, CblasTrans, 50, 70, 30,
                                  1.5f, 2.0f, FALSE, TRUE, 2);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

CTEST(sbgemm_pack, colmajor_pack_a_notrans_notrans)
{
    float err = check_sbgemm_pack(CblasColMajor, CblasNoTrans, CblasNoTrans, 50, 70, 30,
                                  1.5f, 0.0f, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

CTEST(sbgemm_pack, colmajor_pack_both_trans_notrans)
{
    float err = check_sbgemm_pack(CblasColMajor, CblasTrans, CblasNoTrans, 33, 65, 17,
                                  0.75f, 1.0f, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

/* Odd k exercises the bfloat16 pair handling of the copy routines. */
CTEST(sbgemm_pack, rowmajor_pack_b_odd_k)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 33, 17, 1,
                                  1.0f, 0.0f, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
    err = check_sbgemm_pack(CblasRowMajor, CblasTrans, CblasNoTrans, 33, 17, 3,
                            1.0f, 0.0f, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

/* k = 1000 exceeds GEMM_Q on the cores this can run on, so the K loop takes at
 * least two blocks and passes through the halving of the last block. */
CTEST(sbgemm_pack, rowmajor_pack_b_large_k)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 33, 65, 1000,
                                  1.0f, 0.0f, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

CTEST(sbgemm_pack, rowmajor_pack_b_large_n)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasNoTrans, CblasTrans, 40, 1700, 300,
                                  1.0f, 0.0f, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

/* m = 9000 stays inside one GEMM_R panel of the row-major A (the outer
 * operand) on every x86-64 core with the default BUFFER_SIZE. */
CTEST(sbgemm_pack, rowmajor_pack_a_large_m)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 9000, 8, 64,
                                  1.0f, 0.0f, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

/* GEMM_R for sbgemm derives from BUFFER_SIZE and is below 131000 on every
 * x86-64 core with the default 128 MB buffer, so m = 140000 makes the packed
 * outer operand span two GEMM_R panels. */
CTEST(sbgemm_pack, pack_outer_operand_crosses_gemm_r)
{
    float err = check_sbgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 140000, 8, 16,
                                  1.0f, 0.0f, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
    err = check_sbgemm_pack(CblasColMajor, CblasNoTrans, CblasTrans, 8, 140000, 16,
                            1.0f, 0.5f, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
}

/* See test_sgemm_pack.c. Odd k matters for bfloat16, whose copies pair rows. */
CTEST(sbgemm_pack, size_matches_layout_for_every_tail)
{
    blasint m, k;
    float err;

    for (m = 1; m <= 1400; m++) {
        err = check_sbgemm_pack(CblasColMajor, CblasNoTrans, CblasNoTrans, m, 2, 3, 1.0f, 0.0f, TRUE, FALSE, 0);
        ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
        err = check_sbgemm_pack(CblasRowMajor, CblasTrans, CblasNoTrans, m, 2, 3, 1.0f, 0.0f, TRUE, FALSE, 0);
        ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
    }
    for (k = 1; k <= 2200; k++) {
        err = check_sbgemm_pack(CblasColMajor, CblasNoTrans, CblasTrans, 3, 5, k, 1.0f, 0.0f, TRUE, TRUE, 0);
        ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
    }
}

#if defined(__linux__)
CTEST(sbgemm_pack, kernels_stay_inside_get_size)
{
    static const blasint ms[] = {1, 2, 3, 5, 7, 8, 13, 16, 17, 31, 32, 33, 64, 65, 127, 129, 300};
    static const blasint ns[] = {1, 16, 33, 300};
    static const blasint ks[] = {1, 3, 17, 129, 1000};
    size_t mi, ni, ki;
    int order, ta, tb, mode;

    guard_pages = TRUE;
    for (mi = 0; mi < sizeof(ms) / sizeof(ms[0]); mi++)
    for (ni = 0; ni < sizeof(ns) / sizeof(ns[0]); ni++)
    for (ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++)
    for (order = 0; order < 2; order++)
    for (ta = 0; ta < 2; ta++)
    for (tb = 0; tb < 2; tb++)
    for (mode = 0; mode < 3; mode++) {
        float err = check_sbgemm_pack(order ? CblasColMajor : CblasRowMajor,
                                      ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                                      ms[mi], ns[ni], ks[ki], 1.0f, 0.0f,
                                      mode != 1, mode != 0, 0);
        if (err > SINGLE_EPS) {
            guard_pages = FALSE;
            ASSERT_DBL_NEAR_TOL(0.0f, err, SINGLE_EPS);
        }
    }
    guard_pages = FALSE;
}
#endif

CTEST(sbgemm_pack, xerbla_compute_transb_invalid)
{
    bfloat16 a[4] = {0, 0, 0, 0};
    bfloat16 b[4] = {0, 0, 0, 0};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    set_xerbla("SBGEMM_COMPUTE ", 3);
    cblas_sbgemm_compute(CblasColMajor, CblasNoTrans, INVALID, 2, 2, 2, a, 2, b, 2, 0.0f, c, 2);
    ASSERT_TRUE(check_error());
}

/* A buffer packed by cblas_sgemm_pack is not accepted by cblas_sbgemm_compute. */
CTEST(sbgemm_pack, xerbla_compute_rejects_foreign_pack)
{
#ifdef BUILD_SINGLE
    float a[16];
    bfloat16 b[16] = {0};
    float c[16];
    size_t size = cblas_sgemm_pack_get_size(CblasAMatrix, 4, 4, 4);
    float *packed_a = (float *)malloc(size);
    int i;

    for (i = 0; i < 16; i++) a[i] = 1.0f;
    for (i = 0; i < 16; i++) c[i] = 7.0f;
    cblas_sgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 4, 4, 4, 1.0f, a, 4, packed_a);

    set_xerbla("SBGEMM_COMPUTE ", 7);
    cblas_sbgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 4, 4, 4, (bfloat16 *)packed_a, 4,
                         b, 4, 0.0f, c, 4);
    ASSERT_TRUE(check_error());
    for (i = 0; i < 16; i++) ASSERT_DBL_NEAR_TOL(7.0f, c[i], SINGLE_EPS);

    free(packed_a);
#endif
}

/* cblas_sbgemm_status is cblas_sbgemm with a return value, and the packed
 * entry points return one too: 0 when they did their work, the parameter
 * number given to xerbla when an argument or a packed buffer was rejected, and
 * a negative OPENBLAS_GEMM_STATUS_* code for a failure that has no argument to
 * blame. Every value here is exact in bfloat16 and the sums are small, so the
 * status variant, the plain routine and the packed compute agree exactly. */
CTEST(sbgemm_pack, status_reports_success_and_rejections)
{
    bfloat16 a[16], b[16], a5[20];
    float c_ref[16], c[16];
    unsigned char *packed_b;
    int i;

    for (i = 0; i < 16; i++) {
        a[i] = to_bf16(1.0f + (float)(i % 3));
        b[i] = to_bf16(2.0f - (float)(i % 5));
        c_ref[i] = 7.0f;
        c[i] = 7.0f;
    }
    /* A 4 x 5 operand for the k = 5 case: the single precision fallback widens
     * the operand before it reads the packed header, so it must be real. */
    for (i = 0; i < 20; i++) a5[i] = to_bf16(1.0f);

    cblas_sbgemm(CblasColMajor, CblasNoTrans, CblasTrans, 4, 4, 4, 1.0f, a, 4, b, 4, 0.5f, c_ref, 4);
    ASSERT_EQUAL(0, cblas_sbgemm_status(CblasColMajor, CblasNoTrans, CblasTrans, 4, 4, 4, 1.0f, a, 4, b, 4, 0.5f, c, 4));
    for (i = 0; i < 16; i++) ASSERT_DBL_NEAR_TOL(c_ref[i], c[i], SINGLE_EPS);

    ASSERT_EQUAL(0, cblas_sbgemm_status(CblasColMajor, CblasNoTrans, CblasNoTrans, 0, 4, 4, 1.0f, a, 4, b, 4, 0.5f, c, 4));

    set_xerbla("SBGEMM ", 2);
    ASSERT_EQUAL(2, cblas_sbgemm_status(CblasColMajor, CblasNoTrans, INVALID, 4, 4, 4, 1.0f, a, 4, b, 4, 0.0f, c, 4));
    ASSERT_TRUE(check_error());

    set_xerbla("SBGEMM ", 13);
    ASSERT_EQUAL(13, cblas_sbgemm_status(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1.0f, a, 4, b, 4, 0.0f, c, 1));
    ASSERT_TRUE(check_error());

    packed_b = (unsigned char *)malloc(cblas_sbgemm_pack_get_size(CblasBMatrix, 4, 4, 4));
    ASSERT_EQUAL(0, cblas_sbgemm_pack(CblasColMajor, CblasBMatrix, CblasTrans, 4, 4, 4, 1.0f, b, 4, (bfloat16 *)packed_b));
    for (i = 0; i < 16; i++) c[i] = 7.0f;
    ASSERT_EQUAL(0, cblas_sbgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 4, 4, a, 4, (bfloat16 *)packed_b, 4,
                                         0.5f, c, 4));
    for (i = 0; i < 16; i++) ASSERT_DBL_NEAR_TOL(c_ref[i], c[i], SINGLE_EPS);

    /* A rejected argument leaves the buffer untouched, so the earlier pack
     * still serves. */
    set_xerbla("SBGEMM_PACK ", 9);
    ASSERT_EQUAL(9, cblas_sbgemm_pack(CblasColMajor, CblasBMatrix, CblasNoTrans, 4, 4, 4, 1.0f, b, 1, (bfloat16 *)packed_b));
    ASSERT_TRUE(check_error());
    for (i = 0; i < 16; i++) c[i] = 7.0f;
    ASSERT_EQUAL(0, cblas_sbgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 4, 4, a, 4, (bfloat16 *)packed_b, 4,
                                         0.5f, c, 4));
    for (i = 0; i < 16; i++) ASSERT_DBL_NEAR_TOL(c_ref[i], c[i], SINGLE_EPS);

    /* A packed operand made for another k is reported at its own position. */
    set_xerbla("SBGEMM_COMPUTE ", 9);
    ASSERT_EQUAL(9, cblas_sbgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 4, 5, a5, 4, (bfloat16 *)packed_b, 4,
                                         0.0f, c, 4));
    ASSERT_TRUE(check_error());

    /* An order that is neither row nor column major names argument 1. */
    set_xerbla("SBGEMM ", 0);
    ASSERT_EQUAL(1, cblas_sbgemm_status((enum CBLAS_ORDER)INVALID, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1.0f, a, 4, b, 4,
                                        0.0f, c, 4));
    ASSERT_TRUE(check_error());

    free(packed_b);
}

#endif /* NO_CBLAS */
#endif /* BUILD_BFLOAT16 */
