/***************************************************************************
 * Copyright (c) 2025, The OpenBLAS Project
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in
 * the documentation and/or other materials provided with the
 * distribution.
 * 3. Neither the name of the OpenBLAS project nor the names of
 * its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * *****************************************************************************/

#ifndef CBLAS_H
#define CBLAS_H

#include <stddef.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */

/*Set the number of threads on runtime.*/
void openblas_set_num_threads(int num_threads);
void goto_set_num_threads(int num_threads);
int openblas_set_num_threads_local(int num_threads);

/*Get the number of threads on runtime.*/
int openblas_get_num_threads(void);

/*Get the number of physical processors (cores).*/
int openblas_get_num_procs(void);

/*Get the build configure on runtime.*/
char* openblas_get_config(void);

/*Get the CPU corename on runtime.*/
char* openblas_get_corename(void);

/* Per-thread flag, non-zero when an OpenBLAS call in this thread returned
 * without doing its work because memory could not be obtained: no work buffer
 * was free or one could not be mapped, or a temporary of the bfloat16 paths
 * could not be allocated. It is one flag per process where the compiler has
 * no thread-local storage, and in the single-buffer allocator of PPC440
 * builds. The routines that return a status report the same
 * failure through it; the standard void routines, and LAPACK routines whose
 * BLAS calls fail, leave only this flag. Clear it before a call and inspect it
 * after. Set by every routine built from interface/gemm.c, gemm_compute.c,
 * gemv.c, ger.c, symv.c, syr2.c, syr2k.c, trmv.c and trsm.c (so also ?trsm,
 * ?her2k, ?gemm3m and the complex routines those files build), by the bfloat16
 * temporaries of cblas_sbgemm, the packed API and cblas_sbgemv, and by the
 * small-matrix SBGEMM kernel; other routines keep upstream's behaviour. In a
 * threaded build, a failure on a worker thread sets that thread's flag. */
int  openblas_alloc_failed(void);
void openblas_clear_alloc_failed(void);

/*Set the threading backend to a custom callback.*/
typedef void (*openblas_dojob_callback)(int thread_num, void *jobdata, int dojob_data);
typedef void (*openblas_threads_callback)(int sync, openblas_dojob_callback dojob, int numjobs, size_t jobdata_elsize, void *jobdata, int dojob_data);
void openblas_set_threads_callback_function(openblas_threads_callback callback);

/* Replace the XERBLA handler for this OpenBLAS instance and return the
 * previous handler. Passing NULL restores the default. Callbacks may run
 * concurrently and must be thread-safe. The name and info pointers are valid
 * only during the callback; name spans name_length bytes and need not be
 * NUL-terminated. Replacement is thread-safe but does not wait for in-flight
 * calls, so the previous handler must remain loaded until they complete. */
#ifndef OPENBLAS_XERBLA_HANDLER_DEFINED
#define OPENBLAS_XERBLA_HANDLER_DEFINED
typedef void (*openblas_xerbla_handler)(const char *name,
                                        const blasint *info,
                                        size_t name_length);
#endif
openblas_xerbla_handler openblas_set_xerbla(openblas_xerbla_handler handler);

#ifdef OPENBLAS_OS_LINUX
/* Sets thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
int openblas_setaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
/* Queries thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
int openblas_getaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
#endif

/* Get the parallelization type which is used by OpenBLAS */
int openblas_get_parallel(void);
/* OpenBLAS is compiled for sequential use  */
#define OPENBLAS_SEQUENTIAL  0
/* OpenBLAS is compiled using normal threading model */
#define OPENBLAS_THREAD  1
/* OpenBLAS is compiled using OpenMP threading model */
#define OPENBLAS_OPENMP 2


/*
 * Since all of GotoBlas was written without const,
 * we disable it at build time.
 */
#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif


#define CBLAS_INDEX size_t

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;

/* Packed GEMM extension (cblas_?gemm_pack_get_size, cblas_?gemm_pack,
 * cblas_?gemm_compute). CblasPacked is passed in place of a CBLAS_TRANSPOSE
 * value to cblas_?gemm_compute for an operand that was packed beforehand. */
typedef enum CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162} CBLAS_IDENTIFIER;
typedef enum CBLAS_STORAGE    {CblasPacked=151} CBLAS_STORAGE;

/* Status returned by cblas_?gemm_pack, cblas_?gemm_compute, and
 * cblas_sbgemm_status. 0 means the routine did its work. A positive value is
 * the number of the argument that was rejected, which xerbla has also been
 * told. The negative values below name failures that are not tied to one
 * argument. C is not touched when an argument is rejected; after NO_MEMORY it
 * may hold part of the product, and a kernel that fails partway may report only
 * through openblas_alloc_failed(). A failed pack leaves dest
 * untouched for a rejected argument and TOO_LARGE, and otherwise leaves no
 * valid header behind, so that a later cblas_?gemm_compute rejects the buffer. xerbla
 * receives parameter 0 for the first two codes, the other two print a message
 * through openblas_warning instead. */
#define OPENBLAS_GEMM_STATUS_TOO_LARGE (-1) /* the packed size does not fit; cblas_?gemm_pack_get_size returns 0 for these dimensions */
#define OPENBLAS_GEMM_STATUS_INTERNAL  (-2) /* the panels written disagree with the size computed in closed form */
#define OPENBLAS_GEMM_STATUS_NO_MEMORY (-3) /* a work buffer or a temporary could not be obtained; openblas_alloc_failed() is set as well */
#define OPENBLAS_GEMM_STATUS_NO_KERNEL (-4) /* the bfloat16 kernels cannot run in this process and no fallback is compiled in */
	
float  cblas_sdsdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float *y, OPENBLAS_CONST blasint incy);
double cblas_dsdot (OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float *y, OPENBLAS_CONST blasint incy);
float  cblas_sdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);

openblas_complex_float  cblas_cdotu(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy);
openblas_complex_float  cblas_cdotc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy);
openblas_complex_double cblas_zdotu(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy);
openblas_complex_double cblas_zdotc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy);

void  cblas_cdotu_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy, void  *ret);
void  cblas_cdotc_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy, void  *ret);
void  cblas_zdotu_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy, void *ret);
void  cblas_zdotc_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy, void *ret);

float  cblas_sasum (OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
double cblas_dasum (OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
float  cblas_scasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
double cblas_dzasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

float  cblas_ssum (OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
double cblas_dsum (OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
float  cblas_scsum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
double cblas_dzsum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

float  cblas_snrm2 (OPENBLAS_CONST blasint N, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX);
double cblas_dnrm2 (OPENBLAS_CONST blasint N, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX);
float  cblas_scnrm2(OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX);
double cblas_dznrm2(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX);

CBLAS_INDEX cblas_isamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

CBLAS_INDEX cblas_isamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

float cblas_samax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
double cblas_damax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
float cblas_scamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
double cblas_dzamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

float cblas_samin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
double cblas_damin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
float cblas_scamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
double cblas_dzamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

CBLAS_INDEX cblas_ismax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

CBLAS_INDEX cblas_ismin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
void cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
void cblas_caxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
void cblas_zaxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

void cblas_caxpyc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
void cblas_zaxpyc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

void cblas_scopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
void cblas_dcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
void cblas_ccopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
void cblas_zcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

void cblas_sswap(OPENBLAS_CONST blasint n, float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
void cblas_dswap(OPENBLAS_CONST blasint n, double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
void cblas_cswap(OPENBLAS_CONST blasint n, void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
void cblas_zswap(OPENBLAS_CONST blasint n, void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

void cblas_srot(OPENBLAS_CONST blasint N, float *X, OPENBLAS_CONST blasint incX, float *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST float c, OPENBLAS_CONST float s);
void cblas_drot(OPENBLAS_CONST blasint N, double *X, OPENBLAS_CONST blasint incX, double *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST double c, OPENBLAS_CONST double  s);
void cblas_csrot(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST float c, OPENBLAS_CONST float s);
void cblas_zdrot(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST double c, OPENBLAS_CONST double s);

void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_crotg(void *a, void *b, float *c, void *s);
void cblas_zrotg(void *a, void *b, double *c, void *s);


void cblas_srotm(OPENBLAS_CONST blasint N, float *X, OPENBLAS_CONST blasint incX, float *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST float *P);
void cblas_drotm(OPENBLAS_CONST blasint N, double *X, OPENBLAS_CONST blasint incX, double *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST double *P);

void cblas_srotmg(float *d1, float *d2, float *b1, OPENBLAS_CONST float b2, float *P);
void cblas_drotmg(double *d1, double *d2, double *b1, OPENBLAS_CONST double b2, double *P);

void cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, OPENBLAS_CONST blasint incX);
void cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, double *X, OPENBLAS_CONST blasint incX);
void cblas_cscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, void *X, OPENBLAS_CONST blasint incX);
void cblas_zscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, void *X, OPENBLAS_CONST blasint incX);
void cblas_csscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, void *X, OPENBLAS_CONST blasint incX);
void cblas_zdscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, void *X, OPENBLAS_CONST blasint incX);

void cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
void cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
void cblas_cgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST void *beta,  void  *y, OPENBLAS_CONST blasint incy);
void cblas_zgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST void *beta,  void  *y, OPENBLAS_CONST blasint incy);

void cblas_sger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float   alpha, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float  *Y, OPENBLAS_CONST blasint incY, float  *A, OPENBLAS_CONST blasint lda);
void cblas_dger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double  alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A, OPENBLAS_CONST blasint lda);
void cblas_cgeru(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *alpha, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void  *Y, OPENBLAS_CONST blasint incY, void  *A, OPENBLAS_CONST blasint lda);
void cblas_cgerc(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *alpha, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void  *Y, OPENBLAS_CONST blasint incY, void  *A, OPENBLAS_CONST blasint lda);
void cblas_zgeru(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);
void cblas_zgerc(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);

void cblas_strsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
void cblas_dtrsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
void cblas_ctrsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
void cblas_ztrsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

void cblas_strmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
void cblas_dtrmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
void cblas_ctrmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
void cblas_ztrmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

void cblas_ssyr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, float *A, OPENBLAS_CONST blasint lda);
void cblas_dsyr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, double *A, OPENBLAS_CONST blasint lda);
void cblas_cher(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, void *A, OPENBLAS_CONST blasint lda);
void cblas_zher(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, void *A, OPENBLAS_CONST blasint lda);

void cblas_ssyr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo,OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X,
                OPENBLAS_CONST blasint incX, OPENBLAS_CONST float *Y, OPENBLAS_CONST blasint incY, float *A, OPENBLAS_CONST blasint lda);
void cblas_dsyr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X,
                OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A, OPENBLAS_CONST blasint lda);
void cblas_cher2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX,
                OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);
void cblas_zher2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX,
                OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);

void cblas_sgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
void cblas_dgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);
void cblas_cgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
void cblas_zgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);

void cblas_ssbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
void cblas_dsbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);


void cblas_stbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
void cblas_dtbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
void cblas_ctbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
void cblas_ztbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

void cblas_stbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
void cblas_dtbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
void cblas_ctbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
void cblas_ztbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

void cblas_stpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST float *Ap, float *X, OPENBLAS_CONST blasint incX);
void cblas_dtpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST double *Ap, double *X, OPENBLAS_CONST blasint incX);
void cblas_ctpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);
void cblas_ztpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);

void cblas_stpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST float *Ap, float *X, OPENBLAS_CONST blasint incX);
void cblas_dtpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST double *Ap, double *X, OPENBLAS_CONST blasint incX);
void cblas_ctpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);
void cblas_ztpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);

void cblas_ssymv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
void cblas_dsymv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);
void cblas_chemv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
void cblas_zhemv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);


void cblas_sspmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *Ap,
                 OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
void cblas_dspmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *Ap,
                 OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);

void cblas_sspr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, float *Ap);
void cblas_dspr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, double *Ap);

void cblas_chpr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, void *A);
void cblas_zhpr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST void *X,OPENBLAS_CONST blasint incX, void *A);

void cblas_sspr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float *Y, OPENBLAS_CONST blasint incY, float *A);
void cblas_dspr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A);
void cblas_chpr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *Ap);
void cblas_zhpr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *Ap);

void cblas_chbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
void cblas_zhbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);

void cblas_chpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *Ap, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
void cblas_zhpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *Ap, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);

void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_dgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);

/* Packed GEMM: size in bytes of the buffer that holds one packed operand. The buffer needs no alignment beyond
 * that of its element type; the size includes the slack used to align the packed panels internally. */
size_t cblas_sgemm_pack_get_size(OPENBLAS_CONST enum CBLAS_IDENTIFIER identifier, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K);
size_t cblas_dgemm_pack_get_size(OPENBLAS_CONST enum CBLAS_IDENTIFIER identifier, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K);
/* Packed GEMM: pack alpha * op(src) as operand `identifier` of an M x N x K product into dest. */
int cblas_sgemm_pack(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_IDENTIFIER identifier, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		      OPENBLAS_CONST float alpha, OPENBLAS_CONST float *src, OPENBLAS_CONST blasint ld, float *dest);
int cblas_dgemm_pack(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_IDENTIFIER identifier, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		      OPENBLAS_CONST double alpha, OPENBLAS_CONST double *src, OPENBLAS_CONST blasint ld, double *dest);
/* Packed GEMM: C := alpha * op(A) * op(B) + beta * C, where TransA/TransB are CBLAS_TRANSPOSE values or CblasPacked and alpha comes from the packed operand(s). */
int cblas_sgemm_compute(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST blasint TransA, OPENBLAS_CONST blasint TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		         OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
int cblas_dgemm_compute(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST blasint TransA, OPENBLAS_CONST blasint TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		         OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);

void cblas_cgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_cgemm3m(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zgemm3m(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_sgemmt(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_dgemmt(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
void cblas_cgemmt(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zgemmt(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_ssymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_dsymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
void cblas_csymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zsymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_ssyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_dsyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
void cblas_csyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zsyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_ssyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_dsyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
void cblas_csyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zsyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_strmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *B, OPENBLAS_CONST blasint ldb);
void cblas_dtrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *B, OPENBLAS_CONST blasint ldb);
void cblas_ctrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);
void cblas_ztrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);

void cblas_strsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *B, OPENBLAS_CONST blasint ldb);
void cblas_dtrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *B, OPENBLAS_CONST blasint ldb);
void cblas_ctrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);
void cblas_ztrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);

void cblas_chemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zhemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_cherk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                 OPENBLAS_CONST float alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zherk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                 OPENBLAS_CONST double alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_cher2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                  OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, void *C, OPENBLAS_CONST blasint ldc);
void cblas_zher2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                  OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, void *C, OPENBLAS_CONST blasint ldc);

void cblas_xerbla(blasint p, OPENBLAS_CONST char *rout, OPENBLAS_CONST char *form, ...);

/*** BLAS extensions ***/

void cblas_saxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST float beta, float *y, OPENBLAS_CONST blasint incy);

void cblas_daxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST double beta, double *y, OPENBLAS_CONST blasint incy);

void cblas_caxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST void *beta, void *y, OPENBLAS_CONST blasint incy);

void cblas_zaxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST void *beta, void *y, OPENBLAS_CONST blasint incy);

void cblas_somatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, OPENBLAS_CONST float *a, 
		     OPENBLAS_CONST blasint clda, float *b, OPENBLAS_CONST blasint cldb); 
void cblas_domatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double calpha, OPENBLAS_CONST double *a,
		     OPENBLAS_CONST blasint clda, double *b, OPENBLAS_CONST blasint cldb); 
void cblas_comatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float* calpha, OPENBLAS_CONST float* a, 
		     OPENBLAS_CONST blasint clda, float*b, OPENBLAS_CONST blasint cldb); 
void cblas_zomatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double* calpha, OPENBLAS_CONST double* a, 
		     OPENBLAS_CONST blasint clda,  double *b, OPENBLAS_CONST blasint cldb); 

void cblas_simatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, float *a, 
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 
void cblas_dimatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double calpha, double *a,
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 
void cblas_cimatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float* calpha, float* a, 
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 
void cblas_zimatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double* calpha, double* a, 
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 

void cblas_sgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_A,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_C,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, OPENBLAS_CONST float *a, OPENBLAS_CONST blasint clda,OPENBLAS_CONST float cbeta, float *c, 
                  OPENBLAS_CONST blasint cldc);
void cblas_dgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_A,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_C,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double calpha, OPENBLAS_CONST double *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST double cbeta, 
		  double *c, OPENBLAS_CONST blasint cldc); 
void cblas_cgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_A,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_C,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float *calpha, OPENBLAS_CONST float *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST float *cbeta, 
		  float *c, OPENBLAS_CONST blasint cldc); 
void cblas_zgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_A,OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS_C,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double *calpha, OPENBLAS_CONST double *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST double *cbeta, 
		  double *c, OPENBLAS_CONST blasint cldc); 

void cblas_sgemm_batch(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransA_array, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST float * alpha_array, OPENBLAS_CONST float ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST float ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST float * beta_array, float ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void cblas_dgemm_batch(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransA_array, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST double * alpha_array, OPENBLAS_CONST double ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST double ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST double * beta_array, double ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void cblas_cgemm_batch(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransA_array, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST void * alpha_array, OPENBLAS_CONST void ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST void ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST void * beta_array, void ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void cblas_zgemm_batch(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransA_array, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST void * alpha_array, OPENBLAS_CONST void ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST void ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST void * beta_array, void ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void cblas_sgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST float * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST float beta, float * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);

void cblas_dgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST double * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST double beta, double * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);

void cblas_cgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void * alpha, OPENBLAS_CONST void * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST void * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST void * beta, void * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);

void cblas_zgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void * alpha, OPENBLAS_CONST void * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST void * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST void * beta, void * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);

/*** BFLOAT16 and INT8 extensions ***/
/* convert float array to BFLOAT16 array by rounding */
void   cblas_sbstobf16(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *in, OPENBLAS_CONST blasint incin, bfloat16 *out, OPENBLAS_CONST blasint incout);
/* convert double array to BFLOAT16 array by rounding */
void   cblas_sbdtobf16(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *in, OPENBLAS_CONST blasint incin, bfloat16 *out, OPENBLAS_CONST blasint incout);
/* convert BFLOAT16 array to float array */
void   cblas_sbf16tos(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *in, OPENBLAS_CONST blasint incin, float  *out, OPENBLAS_CONST blasint incout);
/* convert BFLOAT16 array to double array */
void   cblas_dbf16tod(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *in, OPENBLAS_CONST blasint incin, double *out, OPENBLAS_CONST blasint incout);
void   cblas_bgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 alpha, OPENBLAS_CONST bfloat16 *a, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST bfloat16 beta, bfloat16 *y, OPENBLAS_CONST blasint incy);
/* dot production of BFLOAT16 input arrays, and output as float */
float  cblas_sbdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST bfloat16 *y, OPENBLAS_CONST blasint incy);
void   cblas_sbgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *a, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float beta, float *y, OPENBLAS_CONST blasint incy);

void cblas_bgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		    OPENBLAS_CONST bfloat16 alpha, OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST bfloat16 beta, bfloat16 *C, OPENBLAS_CONST blasint ldc);
void   cblas_sbgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		    OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
/* cblas_sbgemm with a return value: 0 when the routine did its work, otherwise an
 * OPENBLAS_GEMM_STATUS_* code or the number of the rejected argument (see above). cblas_sbgemm itself keeps the
 * standard void signature and reports through xerbla, openblas_warning and openblas_alloc_failed() only. */
int    cblas_sbgemm_status(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		    OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
/* Packed GEMM for bfloat16 inputs with float accumulation, see cblas_sgemm_pack_get_size, cblas_sgemm_pack, and cblas_sgemm_compute. */
size_t cblas_sbgemm_pack_get_size(OPENBLAS_CONST enum CBLAS_IDENTIFIER identifier, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K);
int    cblas_sbgemm_pack(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_IDENTIFIER identifier, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		         OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *src, OPENBLAS_CONST blasint ld, bfloat16 *dest);
int    cblas_sbgemm_compute(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST blasint TransA, OPENBLAS_CONST blasint TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		            OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_sbgemm_batch(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransA_array, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST float * alpha_array, OPENBLAS_CONST bfloat16 ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST bfloat16 ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST float * beta_array, float ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void cblas_sbgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST bfloat16 * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST float beta, float * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);
/*** FLOAT16 extensions ***/
void cblas_shgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		    OPENBLAS_CONST float alpha, OPENBLAS_CONST hfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST hfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif
