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

#ifdef BUILD_DOUBLE
#ifndef NO_CBLAS

#define PACK_SENTINEL 0x5A
#define PACK_SENTINEL_BYTES 16

/* Packed buffers normally come from malloc with a sentinel tail, so that an
 * overrun by cblas_dgemm_pack is detected. With guard_pages set (Linux only)
 * they come from mmap and end exactly where a PROT_NONE page begins, so that
 * any read or write past cblas_dgemm_pack_get_size(), by the copy routines or
 * by the kernels, faults. */
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

/**
 * Compute the reference with cblas_dgemm, pack the requested operands with
 * cblas_dgemm_pack, run cblas_dgemm_compute, and return the largest absolute
 * difference divided by (1 + largest reference magnitude).
 *
 * When only one operand is packed it records alpha. When both are packed, A
 * records alpha and B records 3, so that the product of the two recorded
 * values, which cblas_dgemm_compute applies, differs from either of them.
 *
 * param order specifies row or column major order
 * param transa specifies op(A), the transposition operation applied to A
 * param transb specifies op(B), the transposition operation applied to B
 * param m - number of rows of op(A) and C
 * param n - number of columns of op(B) and C
 * param k - number of columns of op(A) and rows of op(B)
 * param alpha - scaling factor recorded by the pack (see above)
 * param beta - scaling factor for matrix C
 * param pack_a - non-zero to pack A
 * param pack_b - non-zero to pack B
 * param pad - extra elements added to every leading dimension
 * return scaled largest difference, or a huge value when a buffer was overrun
 */
static double check_dgemm_pack(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa,
                              enum CBLAS_TRANSPOSE transb, blasint m, blasint n, blasint k,
                              double alpha, double beta, int pack_a, int pack_b, blasint pad)
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
    double *a = (double *)malloc((a_size + 1) * sizeof(double));
    double *b = (double *)malloc((b_size + 1) * sizeof(double));
    double *c_ref = (double *)malloc((c_size + 1) * sizeof(double));
    double *c_test = (double *)malloc((c_size + 1) * sizeof(double));
    unsigned char *packed_a = NULL, *packed_b = NULL;
    size_t size_a = 0, size_b = 0;
    double alpha_a = pack_a ? alpha : 1.0;
    double alpha_b = pack_b ? (pack_a ? 3.0 : alpha) : 1.0;
    double ref_alpha = alpha_a * alpha_b;
    double max_diff = 0.0, max_ref = 0.0;
    size_t i;

    drand_generate(a, (blasint)a_size);
    drand_generate(b, (blasint)b_size);
    drand_generate(c_ref, (blasint)c_size);
    for (i = 0; i < c_size; i++) c_test[i] = c_ref[i];

    cblas_dgemm(order, transa, transb, m, n, k, ref_alpha, a, lda, b, ldb, beta, c_ref, ldc);

    if (pack_a) {
        size_a = cblas_dgemm_pack_get_size(CblasAMatrix, m, n, k);
        packed_a = alloc_packed(size_a);
        cblas_dgemm_pack(order, CblasAMatrix, transa, m, n, k, alpha_a, a, lda, (double *)packed_a);
    }
    if (pack_b) {
        size_b = cblas_dgemm_pack_get_size(CblasBMatrix, m, n, k);
        packed_b = alloc_packed(size_b);
        cblas_dgemm_pack(order, CblasBMatrix, transb, m, n, k, alpha_b, b, ldb, (double *)packed_b);
    }

    cblas_dgemm_compute(order, pack_a ? (blasint)CblasPacked : (blasint)transa,
                        pack_b ? (blasint)CblasPacked : (blasint)transb, m, n, k,
                        pack_a ? (double *)packed_a : a, lda,
                        pack_b ? (double *)packed_b : b, ldb, beta, c_test, ldc);

    for (i = 0; i < c_size; i++) {
        double diff = fabs(c_test[i] - c_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (fabs(c_ref[i]) > max_ref) max_ref = fabs(c_ref[i]);
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

    return max_diff / (1.0 + max_ref);
}

/* Fills a 4 x 4 x 4 column-major problem with ones and c with 7, and packs A
 * and B so that the header rejection tests below can pass them in the wrong
 * place. Both packed buffers are valid for (m, n, k) = (4, 4, 4). */
struct dgemm_pack_fixture {
    double a[16];
    double b[16];
    double c[16];
    double *packed_a;
    double *packed_b;
};

static void dgemm_pack_fixture_init(struct dgemm_pack_fixture *f)
{
    int i;
    for (i = 0; i < 16; i++) {
        f->a[i] = 1.0;
        f->b[i] = 1.0;
        f->c[i] = 7.0;
    }
    f->packed_a = (double *)malloc(cblas_dgemm_pack_get_size(CblasAMatrix, 4, 4, 4));
    f->packed_b = (double *)malloc(cblas_dgemm_pack_get_size(CblasBMatrix, 4, 4, 4));
    cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 4, 4, 4, 1.0, f->a, 4, f->packed_a);
    cblas_dgemm_pack(CblasColMajor, CblasBMatrix, CblasNoTrans, 4, 4, 4, 1.0, f->b, 4, f->packed_b);
}

static void dgemm_pack_fixture_free(struct dgemm_pack_fixture *f)
{
    free(f->packed_a);
    free(f->packed_b);
}

static int c_untouched(const double *c)
{
    int i;
    for (i = 0; i < 16; i++)
        if (c[i] != 7.0) return FALSE;
    return TRUE;
}

CTEST(dgemm_pack, get_size_is_positive)
{
    ASSERT_TRUE(cblas_dgemm_pack_get_size(CblasAMatrix, 7, 9, 11) > 0);
    ASSERT_TRUE(cblas_dgemm_pack_get_size(CblasBMatrix, 7, 9, 11) > 0);
    ASSERT_TRUE(cblas_dgemm_pack_get_size(CblasAMatrix, 0, 0, 0) > 0);
}

CTEST(dgemm_pack, rowmajor_pack_b_notrans_notrans)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 50, 70, 30,
                                 1.5, 2.0, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, rowmajor_pack_a_trans_notrans)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasTrans, CblasNoTrans, 50, 70, 30,
                                 0.5, 0.0, TRUE, FALSE, 3);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, rowmajor_pack_both_trans_trans)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasTrans, CblasTrans, 33, 65, 17,
                                 2.0, 1.0, TRUE, TRUE, 1);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, colmajor_pack_b_notrans_trans)
{
    double err = check_dgemm_pack(CblasColMajor, CblasNoTrans, CblasTrans, 50, 70, 30,
                                 1.5, 2.0, FALSE, TRUE, 2);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, colmajor_pack_a_notrans_notrans)
{
    double err = check_dgemm_pack(CblasColMajor, CblasNoTrans, CblasNoTrans, 50, 70, 30,
                                 1.5, 0.0, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, colmajor_pack_both_trans_notrans)
{
    double err = check_dgemm_pack(CblasColMajor, CblasTrans, CblasNoTrans, 33, 65, 17,
                                 0.75, 1.0, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, pack_none_matches_gemm_with_alpha_one)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasTrans, 40, 20, 60,
                                 1.0, 0.5, FALSE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, rowmajor_pack_both_k_1)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 33, 17, 1,
                                 1.0, 0.0, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

/* k = 1000 exceeds GEMM_Q on every core, so the K loop takes at least two
 * blocks and passes through the halving of the last block. */
CTEST(dgemm_pack, rowmajor_pack_b_large_k)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 33, 65, 1000,
                                 1.0, 0.0, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

/* n = 1700 makes the row-major B, the inner operand, span several GEMM_P blocks. */
CTEST(dgemm_pack, rowmajor_pack_b_large_n)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasTrans, 40, 1700, 300,
                                 1.0, 0.0, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

/* m = 1700 makes the column-major A, the inner operand, span several GEMM_P blocks. */
CTEST(dgemm_pack, colmajor_pack_a_large_m)
{
    double err = check_dgemm_pack(CblasColMajor, CblasTrans, CblasNoTrans, 1700, 40, 300,
                                 1.0, 1.0, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

/* The row-major A is the outer operand. m = 9000 stays inside one GEMM_R panel
 * on every x86-64 core with the default BUFFER_SIZE, so this covers a long
 * single panel; the next test crosses the panel boundary. */
CTEST(dgemm_pack, rowmajor_pack_a_large_m)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 9000, 8, 64,
                                 1.0, 0.0, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

/* GEMM_R derives from BUFFER_SIZE and is about 65000 for sgemm with the default
 * 128 MB buffer, so m = 70000 makes the packed outer operand span two GEMM_R
 * panels in row-major order, and the packed B does the same in column-major
 * order. */
CTEST(dgemm_pack, pack_outer_operand_crosses_gemm_r)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 70000, 8, 16,
                                 1.0, 0.0, TRUE, FALSE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
    err = check_dgemm_pack(CblasColMajor, CblasNoTrans, CblasTrans, 8, 70000, 16,
                           1.0, 0.5, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, rowmajor_pack_both_large)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1700, 1700, 300,
                                 1.0, 0.0, TRUE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

CTEST(dgemm_pack, zero_k_scales_c_by_beta)
{
    double err = check_dgemm_pack(CblasRowMajor, CblasNoTrans, CblasNoTrans, 5, 6, 0,
                                 1.0, 0.5, FALSE, TRUE, 0);
    ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
}

/* The packed layout of B does not depend on m, so one packed B serves
 * products with any number of rows. */
CTEST(dgemm_pack, rowmajor_packed_b_reused_for_other_m)
{
    blasint n = 24, k = 40;
    blasint pack_m = 8, sizes[3] = {1, 3, 40};
    size_t size = cblas_dgemm_pack_get_size(CblasBMatrix, pack_m, n, k);
    double *b = (double *)malloc((size_t)k * n * sizeof(double));
    unsigned char *packed_b = alloc_packed(size);
    int s;

    drand_generate(b, k * n);
    cblas_dgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, pack_m, n, k, 1.0, b, n,
                     (double *)packed_b);
    ASSERT_TRUE(sentinel_intact(packed_b, size));

    for (s = 0; s < 3; s++) {
        blasint m = sizes[s];
        double *a = (double *)malloc((size_t)m * k * sizeof(double));
        double *c_ref = (double *)malloc((size_t)m * n * sizeof(double));
        double *c_test = (double *)malloc((size_t)m * n * sizeof(double));
        double max_diff = 0.0;
        blasint i;

        drand_generate(a, m * k);
        for (i = 0; i < m * n; i++) c_ref[i] = c_test[i] = 0.0;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0,
                    c_ref, n);
        cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, m, n, k, a, k,
                            (double *)packed_b, n, 0.0, c_test, n);

        for (i = 0; i < m * n; i++)
            if (fabs(c_test[i] - c_ref[i]) > max_diff) max_diff = fabs(c_test[i] - c_ref[i]);
        ASSERT_DBL_NEAR_TOL(0.0, max_diff, DOUBLE_EPS);

        free(a);
        free(c_ref);
        free(c_test);
    }

    free(b);
    free_packed(packed_b);
}

/* The closed-form size that cblas_dgemm_pack_get_size computes must agree with
 * the panels the pack writes for every tail regime of the M, N, and K blocking:
 * a disagreement makes cblas_dgemm_pack reject the call and leaves the header
 * unwritten, which the compute then rejects, so the product would not match.
 * Column-major A is the inner operand (GEMM_P blocks), row-major A the outer
 * one (GEMM_R blocks); k sweeps the GEMM_Q blocks including the halved tail. */
CTEST(dgemm_pack, size_matches_layout_for_every_tail)
{
    blasint m, k;
    double err;

    for (m = 1; m <= 1400; m++) {
        err = check_dgemm_pack(CblasColMajor, CblasNoTrans, CblasNoTrans, m, 2, 3, 1.0, 0.0, TRUE, FALSE, 0);
        ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
        err = check_dgemm_pack(CblasRowMajor, CblasTrans, CblasNoTrans, m, 2, 3, 1.0, 0.0, TRUE, FALSE, 0);
        ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
    }
    for (k = 1; k <= 2200; k++) {
        err = check_dgemm_pack(CblasColMajor, CblasNoTrans, CblasTrans, 3, 5, k, 1.0, 0.0, TRUE, TRUE, 0);
        ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
    }
}

#if defined(__linux__)
/* Every packed buffer ends exactly at a PROT_NONE page, so a copy routine or a
 * kernel that reads or writes past cblas_dgemm_pack_get_size() faults instead
 * of silently touching neighbouring memory. */
CTEST(dgemm_pack, kernels_stay_inside_get_size)
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
        double err = check_dgemm_pack(order ? CblasColMajor : CblasRowMajor,
                                     ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                                     ms[mi], ns[ni], ks[ki], 1.0, 0.0,
                                     mode != 1, mode != 0, 0);
        if (err > DOUBLE_EPS) {
            guard_pages = FALSE;
            ASSERT_DBL_NEAR_TOL(0.0, err, DOUBLE_EPS);
        }
    }
    guard_pages = FALSE;
}
#endif

/* The header is read and written with memcpy, so a buffer that is only
 * aligned for its element type is accepted. */
CTEST(dgemm_pack, buffer_needs_only_element_alignment)
{
    size_t size = cblas_dgemm_pack_get_size(CblasBMatrix, 6, 10, 12);
    double *storage = (double *)malloc(size + 64);
    double *packed_b = storage + 1;
    double a[6 * 12], b[12 * 10], c_ref[6 * 10], c_test[6 * 10];
    double max_diff = 0.0;
    int i;

    drand_generate(a, 6 * 12);
    drand_generate(b, 12 * 10);
    for (i = 0; i < 60; i++) c_ref[i] = c_test[i] = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 6, 10, 12, 1.0, a, 12, b, 10, 0.0,
                c_ref, 10);
    cblas_dgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, 6, 10, 12, 1.0, b, 10, packed_b);
    cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, 6, 10, 12, a, 12, packed_b, 10,
                        0.0, c_test, 10);

    for (i = 0; i < 60; i++)
        if (fabs(c_test[i] - c_ref[i]) > max_diff) max_diff = fabs(c_test[i] - c_ref[i]);
    ASSERT_DBL_NEAR_TOL(0.0, max_diff, DOUBLE_EPS);

    free(storage);
}

CTEST(dgemm_pack, xerbla_get_size_negative_k)
{
    set_xerbla("DGEMM_PACK_GET_SIZE ", 4);
    ASSERT_TRUE(cblas_dgemm_pack_get_size(CblasAMatrix, 2, 2, -1) == 0);
    ASSERT_TRUE(check_error());
}

/* Dimensions whose packed size does not fit are reported as parameter 0 and
 * answered with 0, quickly: the size is computed in closed form. */
CTEST(dgemm_pack, xerbla_get_size_overflow)
{
    blasint huge = (blasint)(((blasint)1 << (sizeof(blasint) * 8 - 2)) - 1) * 2 + 1;
    double a[1];
    double dest[64];

    set_xerbla("DGEMM_PACK_GET_SIZE ", 0);
    ASSERT_TRUE(cblas_dgemm_pack_get_size(CblasAMatrix, huge, 1, huge) == 0);
    ASSERT_TRUE(check_error());

    set_xerbla("DGEMM_PACK ", 0);
    cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, huge, 1, huge, 1.0, a, huge, dest);
    ASSERT_TRUE(check_error());
}

CTEST(dgemm_pack, xerbla_pack_src_null)
{
    double dest[64];

    set_xerbla("DGEMM_PACK ", 8);
    cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 2, 2, 2, 1.0, NULL, 2, dest);
    ASSERT_TRUE(check_error());
}

CTEST(dgemm_pack, xerbla_compute_operand_null)
{
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double b[4] = {0.0, 0.0, 0.0, 0.0};
    double c[4] = {0.0, 0.0, 0.0, 0.0};

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, NULL, 2, b, 2, 0.0, c, 2);
    ASSERT_TRUE(check_error());

    set_xerbla("DGEMM_COMPUTE ", 9);
    cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, a, 2, NULL, 2, 0.0, c, 2);
    ASSERT_TRUE(check_error());
}

/* A header whose data offset disagrees with the buffer address is rejected;
 * this also catches a packed buffer copied to a differently aligned address. */
CTEST(dgemm_pack, xerbla_compute_rejects_shifted_data_offset)
{
    struct dgemm_pack_fixture f;
    size_t size = cblas_dgemm_pack_get_size(CblasAMatrix, 4, 4, 4);
    unsigned char *copy = (unsigned char *)malloc(size + 64);
    unsigned char *shifted;
    int i;

    dgemm_pack_fixture_init(&f);
    /* Move the bytes to an address with a different alignment modulo 64. */
    shifted = copy;
    while ((uintptr_t)shifted % 64 == (uintptr_t)f.packed_a % 64) shifted += 16;
    memcpy(shifted, f.packed_a, size);

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 4, 4, 4, (double *)shifted, 4, f.b, 4,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    for (i = 0; i < 16; i++) ASSERT_DBL_NEAR_TOL(7.0, f.c[i], DOUBLE_EPS);

    free(copy);
    dgemm_pack_fixture_free(&f);
}

CTEST(dgemm_pack, xerbla_pack_identifier_invalid)
{
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double dest[64];

    set_xerbla("DGEMM_PACK ", 2);
    cblas_dgemm_pack(CblasColMajor, (enum CBLAS_IDENTIFIER)INVALID, CblasNoTrans, 2, 2, 2, 1.0,
                     a, 2, dest);
    ASSERT_TRUE(check_error());
}

CTEST(dgemm_pack, xerbla_pack_trans_packed_invalid)
{
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double dest[64];

    set_xerbla("DGEMM_PACK ", 3);
    cblas_dgemm_pack(CblasColMajor, CblasAMatrix, (enum CBLAS_TRANSPOSE)CblasPacked, 2, 2, 2, 1.0,
                     a, 2, dest);
    ASSERT_TRUE(check_error());
}

CTEST(dgemm_pack, xerbla_pack_ld_invalid)
{
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double dest[64];

    set_xerbla("DGEMM_PACK ", 9);
    cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 2, 2, 2, 1.0, a, 1, dest);
    ASSERT_TRUE(check_error());
}

CTEST(dgemm_pack, xerbla_compute_transa_invalid)
{
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double b[4] = {0.0, 0.0, 0.0, 0.0};
    double c[4] = {0.0, 0.0, 0.0, 0.0};

    set_xerbla("DGEMM_COMPUTE ", 2);
    cblas_dgemm_compute(CblasColMajor, INVALID, CblasNoTrans, 2, 2, 2, a, 2, b, 2, 0.0, c, 2);
    ASSERT_TRUE(check_error());
}

CTEST(dgemm_pack, xerbla_compute_ldc_invalid)
{
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double b[4] = {0.0, 0.0, 0.0, 0.0};
    double c[4] = {0.0, 0.0, 0.0, 0.0};

    set_xerbla("DGEMM_COMPUTE ", 13);
    cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, a, 2, b, 2, 0.0, c, 1);
    ASSERT_TRUE(check_error());
}

/* A packed buffer made for another k is rejected before C is touched, and the
 * offending operand is reported at the user's argument position in both
 * storage orders. */
CTEST(dgemm_pack, xerbla_compute_rejects_mismatched_k)
{
    struct dgemm_pack_fixture f;
    double b5[20], a5[20];
    int i;

    dgemm_pack_fixture_init(&f);
    for (i = 0; i < 20; i++) a5[i] = b5[i] = 1.0;

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 4, 4, 5, f.packed_a, 4, b5, 5,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    set_xerbla("DGEMM_COMPUTE ", 9);
    cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 4, 5, a5, 4, f.packed_b, 4,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    /* In row-major order the user's B is the driver's A and vice versa; the
     * buffers below were packed row-major so that the identifiers match. */
    cblas_dgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, 4, 4, 4, 1.0, f.a, 4, f.packed_a);
    cblas_dgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, 4, 4, 4, 1.0, f.b, 4, f.packed_b);

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, 4, 4, 5, f.packed_a, 4, b5, 4,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    set_xerbla("DGEMM_COMPUTE ", 9);
    cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, 4, 4, 5, a5, 5, f.packed_b, 4,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    dgemm_pack_fixture_free(&f);
}

/* A buffer packed as A is not accepted in the B position, and vice versa. */
CTEST(dgemm_pack, xerbla_compute_rejects_swapped_identifier)
{
    struct dgemm_pack_fixture f;

    dgemm_pack_fixture_init(&f);

    set_xerbla("DGEMM_COMPUTE ", 9);
    cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 4, 4, f.a, 4, f.packed_a, 4,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 4, 4, 4, f.packed_b, 4, f.b, 4,
                        0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    dgemm_pack_fixture_free(&f);
}

/* The extent of the packed operand itself must match; the other extent may
 * differ (see rowmajor_packed_b_reused_for_other_m). */
CTEST(dgemm_pack, xerbla_compute_rejects_mismatched_own_extent)
{
    struct dgemm_pack_fixture f;
    double b_wide[4 * 5];
    double c_wide[4 * 5];
    int i;

    dgemm_pack_fixture_init(&f);
    for (i = 0; i < 20; i++) {
        b_wide[i] = 1.0;
        c_wide[i] = 7.0;
    }

    /* Packed A is 4 x 4; asking for m = 5 rows of op(A) must fail. */
    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 5, 4, 4, f.packed_a, 5, f.b, 4,
                        0.0, c_wide, 5);
    ASSERT_TRUE(check_error());
    for (i = 0; i < 20; i++) ASSERT_DBL_NEAR_TOL(7.0, c_wide[i], DOUBLE_EPS);

    /* Packed B has 4 columns; asking for n = 5 must fail, while a different m
     * with the same packed B is fine. */
    set_xerbla("DGEMM_COMPUTE ", 9);
    cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 5, 4, f.a, 4, f.packed_b, 4,
                        0.0, c_wide, 4);
    ASSERT_TRUE(check_error());
    for (i = 0; i < 20; i++) ASSERT_DBL_NEAR_TOL(7.0, c_wide[i], DOUBLE_EPS);

    (void)b_wide;
    dgemm_pack_fixture_free(&f);
}

/* A zero-filled buffer has no valid header. */
CTEST(dgemm_pack, xerbla_compute_rejects_zeroed_buffer)
{
    struct dgemm_pack_fixture f;
    size_t size = cblas_dgemm_pack_get_size(CblasAMatrix, 4, 4, 4);
    double *zeroed = (double *)calloc(size, 1);

    dgemm_pack_fixture_init(&f);

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 4, 4, 4, zeroed, 4, f.b, 4, 0.0,
                        f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    free(zeroed);
    dgemm_pack_fixture_free(&f);
}

/* When both packed operands are unusable, the lower argument (a, 7) is
 * reported in both storage orders. */
CTEST(dgemm_pack, xerbla_compute_reports_a_when_both_invalid)
{
    struct dgemm_pack_fixture f;

    dgemm_pack_fixture_init(&f);

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasPacked, 4, 4, 5, f.packed_a, 4,
                        f.packed_b, 4, 0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasRowMajor, CblasPacked, CblasPacked, 4, 4, 5, f.packed_a, 4,
                        f.packed_b, 4, 0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    dgemm_pack_fixture_free(&f);
}

#ifdef BUILD_SINGLE
/* A buffer packed by cblas_sgemm_pack is not accepted by cblas_dgemm_compute. */
CTEST(dgemm_pack, xerbla_compute_rejects_foreign_pack)
{
    struct dgemm_pack_fixture f;
    float a[16];
    float *foreign = (float *)malloc(cblas_sgemm_pack_get_size(CblasAMatrix, 4, 4, 4));
    int i;

    dgemm_pack_fixture_init(&f);
    for (i = 0; i < 16; i++) a[i] = 1.0f;
    cblas_sgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 4, 4, 4, 1.0f, a, 4, foreign);

    set_xerbla("DGEMM_COMPUTE ", 7);
    cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasNoTrans, 4, 4, 4, (double *)foreign, 4, f.b,
                        4, 0.0, f.c, 4);
    ASSERT_TRUE(check_error());
    ASSERT_TRUE(c_untouched(f.c));

    free(foreign);
    dgemm_pack_fixture_free(&f);
}
#endif

/* See test_sgemm_pack.c. */
CTEST(dgemm_pack, status_reports_success_and_rejections)
{
    struct dgemm_pack_fixture f;
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double dest[64];
    blasint huge = (blasint)(((blasint)1 << (sizeof(blasint) * 8 - 2)) - 1) * 2 + 1;

    dgemm_pack_fixture_init(&f);

    ASSERT_EQUAL(0, cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 4, 4, 4, 1.0, f.a, 4, f.packed_a));
    ASSERT_EQUAL(0, cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasPacked, 4, 4, 4, f.packed_a, 4,
                                        f.packed_b, 4, 0.0, f.c, 4));
    ASSERT_EQUAL(0, cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasNoTrans, 0, 4, 4, f.a, 4, f.b, 4,
                                        0.0, f.c, 4));

    set_xerbla("DGEMM_PACK ", 9);
    ASSERT_EQUAL(9, cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, 2, 2, 2, 1.0, a, 1, dest));
    ASSERT_TRUE(check_error());

    set_xerbla("DGEMM_PACK ", 0);
    ASSERT_EQUAL(OPENBLAS_GEMM_STATUS_TOO_LARGE,
                 cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, huge, 1, huge, 1.0, a, huge, dest));
    ASSERT_TRUE(check_error());

    set_xerbla("DGEMM_COMPUTE ", 13);
    ASSERT_EQUAL(13, cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, a, 2, a, 2, 0.0,
                                         dest, 1));
    ASSERT_TRUE(check_error());

    set_xerbla("DGEMM_COMPUTE ", 9);
    ASSERT_EQUAL(9, cblas_dgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, 4, 4, 5, f.a, 4, f.packed_b, 4,
                                        0.0, f.c, 4));
    ASSERT_TRUE(check_error());

    dgemm_pack_fixture_free(&f);
}

#endif /* NO_CBLAS */
#endif /* BUILD_DOUBLE */
