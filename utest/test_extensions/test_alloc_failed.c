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

#ifndef NO_CBLAS

/* The per-thread allocation failure flag: clear at the start, still clear
 * after calls that obtained their buffers, and set by the internal note. A
 * real failure needs the process to be short of memory, which a unit test
 * cannot arrange portably; the failure-injection harness described in the
 * fork's design note (change/openblas-alloc-failure.md) does that with a job
 * object on Windows. */
CTEST(alloc_failed, flag_reflects_calls)
{
#ifdef BUILD_SINGLE
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float c[4] = {7.0f, 7.0f, 7.0f, 7.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};
    int i;

    openblas_clear_alloc_failed();
    ASSERT_EQUAL(0, openblas_alloc_failed());

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2);
    ASSERT_EQUAL(0, openblas_alloc_failed());
    for (i = 0; i < 4; i++) ASSERT_DBL_NEAR_TOL(a[i], c[i], SINGLE_EPS);

    cblas_sgemv(CblasColMajor, CblasNoTrans, 2, 2, 1.0f, a, 2, x, 1, 0.0f, y, 1);
    ASSERT_EQUAL(0, openblas_alloc_failed());
    ASSERT_DBL_NEAR_TOL(4.0f, y[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(6.0f, y[1], SINGLE_EPS);

    blas_memory_note_failure();
    ASSERT_EQUAL(1, openblas_alloc_failed());
    openblas_clear_alloc_failed();
    ASSERT_EQUAL(0, openblas_alloc_failed());
#endif
}

#endif /* NO_CBLAS */
