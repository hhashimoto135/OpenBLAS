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
 * Driver for the MKL-style packed GEMM extension:
 *
 *   ?gemm_packed_size     bytes needed for one packed operand
 *   ?gemm_packed_pack     pack op(A) or op(B) into the blocked layout
 *   ?gemm_packed_compute  GEMM where either operand may be pre-packed
 *
 * The packed layout is exactly the sequence of panels that the plain GEMM
 * driver (level3.c) would have produced with GEMM_ITCOPY/GEMM_INCOPY for the
 * inner operand ("A", blocked by GEMM_P along m and GEMM_Q along k) or with
 * GEMM_ONCOPY/GEMM_OTCOPY for the outer operand ("B", blocked by GEMM_R along
 * n and GEMM_Q along k). The compute loop below mirrors level3.c and merely
 * substitutes a pointer into the pre-packed panels for the copy step, so the
 * micro-kernel sees the same data as in a regular GEMM.
 *
 * Every block boundary is decided by gemm_packed_k_block(),
 * gemm_packed_m_block(), and gemm_packed_n_block(). Both the pack and the
 * compute side walk the same loops with the same helpers, which is what keeps
 * the offsets consistent. Under DYNAMIC_ARCH the GEMM_P/Q/R, GEMM_UNROLL_*
 * and the copy/kernel entry points all resolve through the gotoblas table, so
 * this file is compiled once and follows whatever core was selected at run
 * time.
 *
 * This file is compiled once per supported type (SINGLE, DOUBLE, BFLOAT16)
 * and takes its function names from CNAME (sgemm_packed, dgemm_packed,
 * sbgemm_packed).
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "common.h"
#include "common_sbfallback.h"

#if defined(COMPLEX) || defined(XDOUBLE) || defined(HFLOAT16) || defined(BGEMM)
#error "gemm_packed.c supports SGEMM, DGEMM, and SBGEMM only"
#endif

#define GEMM_PACKED_CONCAT_(a, b) a##b
#define GEMM_PACKED_CONCAT(a, b) GEMM_PACKED_CONCAT_(a, b)

#define GEMM_PACKED_SIZE_FN    GEMM_PACKED_CONCAT(CNAME, _size)
#define GEMM_PACKED_PACK_FN    GEMM_PACKED_CONCAT(CNAME, _pack)
#define GEMM_PACKED_COMPUTE_FN GEMM_PACKED_CONCAT(CNAME, _compute)

#if defined(BFLOAT16)
#define GEMM_PACKED_TYPE_TAG GEMM_PACKED_TAG_SB
#elif defined(DOUBLE)
#define GEMM_PACKED_TYPE_TAG GEMM_PACKED_TAG_D
#else
#define GEMM_PACKED_TYPE_TAG GEMM_PACKED_TAG_S
#endif

#if defined(BFLOAT16)
#if defined(DYNAMIC_ARCH)
#define BFLOAT16_ALIGN_K gotoblas->sbgemm_align_k
#else
#define BFLOAT16_ALIGN_K SBGEMM_ALIGN_K
#endif
#endif

/* Header stored in the first GEMM_PACKED_HEADER_BYTES of every packed buffer.
 * All fields are in the internal column-major orientation used by the driver,
 * i.e. after the CBLAS row-major swap has been applied by the interface layer.
 * The header is copied in and out with memcpy, so the caller's buffer needs no
 * alignment beyond that of its element type; only the panel data behind the
 * header is placed on a GEMM_PACKED_ALIGN boundary. */
typedef struct {
  uint32_t magic;
  uint32_t type_tag;
  uint32_t version;
  uint32_t identifier;
  BLASLONG m;
  BLASLONG n;
  BLASLONG k;
  BLASLONG gemm_p;
  BLASLONG gemm_q;
  BLASLONG gemm_r;
  BLASLONG unroll_m;
  BLASLONG unroll_n;
  BLASLONG align_k;
  size_t   data_offset;
  size_t   data_size;
  double   alpha;
} gemm_packed_header_t;

/* The header must fit in the reserved region. */
typedef char gemm_packed_header_fits[(sizeof(gemm_packed_header_t) <= GEMM_PACKED_HEADER_BYTES) ? 1 : -1];

static inline BLASLONG gemm_packed_align_k(void) {
#if defined(BFLOAT16)
  return (BLASLONG)BFLOAT16_ALIGN_K;
#else
  return 1;
#endif
}

/* K blocking, identical to the ls loop of level3.c. */
static inline BLASLONG gemm_packed_k_block(BLASLONG remaining) {
  BLASLONG min_l = remaining;
  if (min_l >= GEMM_Q * 2) {
    min_l = GEMM_Q;
  } else if (min_l > GEMM_Q) {
    min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
  }
  return min_l;
}

/* M blocking, identical to the is loop of level3.c. */
static inline BLASLONG gemm_packed_m_block(BLASLONG remaining) {
  BLASLONG min_i = remaining;
  if (min_i >= GEMM_P * 2) {
    min_i = GEMM_P;
  } else if (min_i > GEMM_P) {
    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
  }
  return min_i;
}

/* N blocking, identical to the js loop of level3.c. */
static inline BLASLONG gemm_packed_n_block(BLASLONG remaining) {
  return remaining > GEMM_R ? GEMM_R : remaining;
}

/* Padded K extent of one panel. Only BFLOAT16 kernels pad K. */
static inline BLASLONG gemm_packed_pad_k(BLASLONG min_l) {
  BLASLONG align = gemm_packed_align_k();
  return (min_l + align - 1) & ~(align - 1);
}

static inline size_t gemm_packed_align_up(size_t bytes) {
  return (bytes + GEMM_PACKED_ALIGN - 1) & ~((size_t)GEMM_PACKED_ALIGN - 1);
}

static inline char *gemm_packed_align_ptr(char *p) {
  uintptr_t v = (uintptr_t)p;
  v = (v + GEMM_PACKED_ALIGN - 1) & ~((uintptr_t)GEMM_PACKED_ALIGN - 1);
  return (char *)v;
}

/* Checked size_t arithmetic. Sizes are capped at PTRDIFF_MAX, the largest
 * object a pointer difference can span. Each returns 0 on overflow. */
static inline int gemm_packed_add(size_t a, size_t b, size_t *out) {
  if (a > (size_t)PTRDIFF_MAX || b > (size_t)PTRDIFF_MAX - a) return 0;
  *out = a + b;
  return 1;
}

static inline int gemm_packed_mul(size_t a, size_t b, size_t *out) {
  if (b != 0 && a > (size_t)PTRDIFF_MAX / b) return 0;
  *out = a * b;
  return 1;
}

/* Bytes reserved for one packed panel of pad_k rows and extent columns. The
 * column count is rounded up to the register-block width so that a copy
 * routine which pads its tail never runs past the reservation. Returns 0 on
 * overflow. */
static inline int gemm_packed_panel_bytes_checked(BLASLONG pad_k, BLASLONG extent, BLASLONG unroll,
                                                  size_t *bytes) {
  size_t padded = (((size_t)extent + (size_t)unroll - 1) / (size_t)unroll) * (size_t)unroll;
  size_t value;
  if (!gemm_packed_mul((size_t)pad_k, padded, &value)) return 0;
  if (!gemm_packed_mul(value, sizeof(IFLOAT), &value)) return 0;
  if (!gemm_packed_add(value, GEMM_PACKED_ALIGN - 1, &value)) return 0;
  *bytes = value & ~((size_t)GEMM_PACKED_ALIGN - 1);
  return 1;
}

/* Unchecked form for the pack and compute loops, whose inputs have already
 * passed ?gemm_packed_size() or the header check. */
static inline size_t gemm_packed_panel_bytes(BLASLONG pad_k, BLASLONG extent, BLASLONG unroll) {
  size_t bytes = 0;
  (void)gemm_packed_panel_bytes_checked(pad_k, extent, unroll, &bytes);
  return bytes;
}

/* The block sizes that gemm_packed_k_block() or gemm_packed_m_block() produce
 * for one dimension, without walking the loop: `full_count` blocks of `block`
 * followed by up to two tail blocks. */
typedef struct {
  size_t   full_count;
  BLASLONG block;
  BLASLONG tail[2];
  int      tail_count;
} gemm_packed_split_t;

static void gemm_packed_split(BLASLONG extent, BLASLONG block, BLASLONG unroll, gemm_packed_split_t *out) {
  BLASLONG remaining = extent;

  out->block = block;
  out->full_count = 0;
  out->tail_count = 0;
  if (remaining >= 2 * block) {
    /* The loop takes `block` while at least 2 * block remain, which leaves a
     * remainder in [block, 2 * block). */
    out->full_count = (size_t)(remaining / block) - 1;
    remaining -= (BLASLONG)out->full_count * block;
  }
  if (remaining > block) {
    BLASLONG half = ((remaining / 2 + unroll - 1) / unroll) * unroll;
    out->tail[0] = half;
    out->tail[1] = remaining - half;
    out->tail_count = 2;
  } else if (remaining > 0) {
    out->tail[0] = remaining;
    out->tail_count = 1;
  }
}

/* Bytes of all M panels for one K block of pad_k rows: the inner operand. */
static int gemm_packed_a_row_bytes(BLASLONG pad_k, const gemm_packed_split_t *m_split, size_t *bytes) {
  size_t total, part;
  int i;

  if (!gemm_packed_panel_bytes_checked(pad_k, m_split->block, GEMM_UNROLL_M, &part)) return 0;
  if (!gemm_packed_mul(part, m_split->full_count, &total)) return 0;
  for (i = 0; i < m_split->tail_count; i++) {
    if (!gemm_packed_panel_bytes_checked(pad_k, m_split->tail[i], GEMM_UNROLL_M, &part)) return 0;
    if (!gemm_packed_add(total, part, &total)) return 0;
  }
  *bytes = total;
  return 1;
}

/* Bytes of all N panels for one K block of pad_k rows: the outer operand. */
static int gemm_packed_b_row_bytes(BLASLONG pad_k, BLASLONG n, size_t *bytes) {
  size_t total, part;

  if (!gemm_packed_panel_bytes_checked(pad_k, GEMM_R, GEMM_UNROLL_N, &part)) return 0;
  if (!gemm_packed_mul(part, (size_t)(n / GEMM_R), &total)) return 0;
  if (n % GEMM_R) {
    if (!gemm_packed_panel_bytes_checked(pad_k, n % GEMM_R, GEMM_UNROLL_N, &part)) return 0;
    if (!gemm_packed_add(total, part, &total)) return 0;
  }
  *bytes = total;
  return 1;
}

/* Data bytes of the packed operand, in closed form: the sum over the K blocks
 * (full blocks and up to two tails) of the panel bytes of that block. This is
 * exactly what the pack loop writes; the pack verifies that after the fact.
 * Returns 0 on overflow. */
static int gemm_packed_data_bytes(int identifier, BLASLONG extent, BLASLONG k, size_t *bytes) {
  gemm_packed_split_t k_split, m_split;
  size_t total = 0, row, part;
  int i;

  if (extent == 0 || k == 0) {
    *bytes = 0;
    return 1;
  }

  gemm_packed_split(k, GEMM_Q, GEMM_UNROLL_M, &k_split);
  gemm_packed_split(extent, GEMM_P, GEMM_UNROLL_M, &m_split);

  if (identifier == GEMM_PACKED_IDENTIFIER_A) {
    if (!gemm_packed_a_row_bytes(gemm_packed_pad_k(k_split.block), &m_split, &row)) return 0;
  } else {
    if (!gemm_packed_b_row_bytes(gemm_packed_pad_k(k_split.block), extent, &row)) return 0;
  }
  if (!gemm_packed_mul(row, k_split.full_count, &total)) return 0;

  for (i = 0; i < k_split.tail_count; i++) {
    if (identifier == GEMM_PACKED_IDENTIFIER_A) {
      if (!gemm_packed_a_row_bytes(gemm_packed_pad_k(k_split.tail[i]), &m_split, &part)) return 0;
    } else {
      if (!gemm_packed_b_row_bytes(gemm_packed_pad_k(k_split.tail[i]), extent, &part)) return 0;
    }
    if (!gemm_packed_add(total, part, &total)) return 0;
  }

  *bytes = total;
  return 1;
}

static inline int gemm_packed_total_bytes(size_t data_bytes, size_t *bytes) {
  size_t total = GEMM_PACKED_HEADER_BYTES + (GEMM_PACKED_ALIGN - 1) + GEMM_PACKED_TAIL_GUARD;
  return gemm_packed_add(total, data_bytes, bytes);
}

/*
 * Bytes needed to pack an operand with `extent` rows or columns of its own
 * (m for A, n for B) against a K dimension of `k`. The result covers both the
 * inner and the outer layout, because the CBLAS row-major convention swaps the
 * operands and the caller of ?gemm_pack_get_size does not tell us the order.
 * Returns 0 when a dimension is negative or the size does not fit.
 */
size_t GEMM_PACKED_SIZE_FN(BLASLONG extent, BLASLONG k) {
  size_t a_bytes, b_bytes, total;

  if (extent < 0 || k < 0) return 0;

#ifdef SBGEMM_FLOAT_FALLBACK
  /* The buffer then holds the float panels that the SGEMM copy routines
   * write, so it is the SGEMM layout that has to fit. */
  if (sbgemm_float_fallback()) return sgemm_packed_size(extent, k);
#endif

  if (!gemm_packed_data_bytes(GEMM_PACKED_IDENTIFIER_A, extent, k, &a_bytes)) return 0;
  if (!gemm_packed_data_bytes(GEMM_PACKED_IDENTIFIER_B, extent, k, &b_bytes)) return 0;
  if (!gemm_packed_total_bytes(a_bytes > b_bytes ? a_bytes : b_bytes, &total)) return 0;
  return total;
}

/*
 * Packs one operand.
 *
 *   identifier  GEMM_PACKED_IDENTIFIER_A (inner, m x k) or
 *               GEMM_PACKED_IDENTIFIER_B (outer, k x n)
 *   trans       0 when the operand is stored as given, 1 when op() transposes
 *   m, n, k     dimensions of the product, internal orientation
 *   alpha       scalar recorded in the header and applied at compute time
 *   src, ld     source matrix in column-major storage
 *   dest        buffer of at least ?gemm_packed_size() bytes
 *
 * Returns 0 on success, 1 for an invalid identifier, 2 when the packed size
 * does not fit (so that ?gemm_packed_size() had returned 0), 3 when the
 * panels written disagree with the size computed in closed form, which would
 * be an internal error, and 4 when the float expansion of the single
 * precision SBGEMM fallback could not be allocated.
 */
int GEMM_PACKED_PACK_FN(int identifier, int trans, BLASLONG m, BLASLONG n, BLASLONG k,
                        FLOAT alpha, IFLOAT *src, BLASLONG ld, void *dest) {
  gemm_packed_header_t header;
  char *data = gemm_packed_align_ptr((char *)dest + GEMM_PACKED_HEADER_BYTES);
  char *cursor = data;
  size_t expected, total;
  BLASLONG ls, is, js, jjs, min_l, min_i, min_j, min_jj, pad_l;

  if (identifier != GEMM_PACKED_IDENTIFIER_A && identifier != GEMM_PACKED_IDENTIFIER_B) return 1;

#ifdef SBGEMM_FLOAT_FALLBACK
  /* Pack the float expansion with the SGEMM copy routines, so that the panels
   * match the kernels ?gemm_packed_compute() will run, see
   * common_sbfallback.h. The header is then written by sgemm_packed_pack and
   * carries the SGEMM type tag and blocking parameters. */
  if (sbgemm_float_fallback()) {
    BLASLONG rows = (identifier == GEMM_PACKED_IDENTIFIER_A) ? (trans ? k : m) : (trans ? n : k);
    BLASLONG cols = (identifier == GEMM_PACKED_IDENTIFIER_A) ? (trans ? m : k) : (trans ? k : n);
    float *src_float = sbgemm_expand_to_float(src, rows, cols, ld);
    int fallback_status;

    if (src_float == NULL) {
      /* Leave no valid header behind, so that a later ?gemm_packed_compute()
       * rejects the buffer instead of reading stale panels. */
      memset(dest, 0, GEMM_PACKED_HEADER_BYTES);
      openblas_warning(0, SBGEMM_FALLBACK_NO_MEMORY);
      return 4;
    }

    fallback_status = sgemm_packed_pack(identifier, trans, m, n, k, alpha, src_float,
                                        sbgemm_expanded_ld(rows), dest);
    free(src_float);
    if (fallback_status == 0) {
      /* The panels are SGEMM's, but the buffer is an SBGEMM one and only
       * cblas_sbgemm_compute may consume it. Stamp our tag over the one
       * sgemm_packed_pack wrote; the compute side asks for it by name. */
      uint32_t tag = GEMM_PACKED_TYPE_TAG;
      memcpy((char *)dest + offsetof(gemm_packed_header_t, type_tag), &tag, sizeof(tag));
    }
    return fallback_status;
  }
#endif

  if (!gemm_packed_data_bytes(identifier, identifier == GEMM_PACKED_IDENTIFIER_A ? m : n, k, &expected) ||
      !gemm_packed_total_bytes(expected, &total)) return 2;

  if (identifier == GEMM_PACKED_IDENTIFIER_A) {
    /* Mirrors ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa) of level3.c. */
    for (ls = 0; ls < k; ls += min_l) {
      min_l = gemm_packed_k_block(k - ls);
      pad_l = gemm_packed_pad_k(min_l);
      for (is = 0; is < m; is += min_i) {
        min_i = gemm_packed_m_block(m - is);
        if (trans) {
          GEMM_INCOPY(min_l, min_i, src + (ls + is * ld), ld, (IFLOAT *)cursor);
        } else {
          GEMM_ITCOPY(min_l, min_i, src + (is + ls * ld), ld, (IFLOAT *)cursor);
        }
        cursor += gemm_packed_panel_bytes(pad_l, min_i, GEMM_UNROLL_M);
      }
    }
  } else {
    /* Mirrors the jjs loop of level3.c with l1stride == 1, so that every
     * GEMM_R-wide panel is complete and can be consumed by one kernel call. */
    for (js = 0; js < n; js += min_j) {
      min_j = gemm_packed_n_block(n - js);
      for (ls = 0; ls < k; ls += min_l) {
        min_l = gemm_packed_k_block(k - ls);
        pad_l = gemm_packed_pad_k(min_l);
        for (jjs = js; jjs < js + min_j; jjs += min_jj) {
          IFLOAT *panel = (IFLOAT *)cursor + pad_l * (jjs - js);
          min_jj = min_j + js - jjs;
#if defined(SKYLAKEX) || defined(COOPERLAKE) || defined(SAPPHIRERAPIDS)
          if (min_jj >= 6 * GEMM_UNROLL_N) min_jj = 6 * GEMM_UNROLL_N;
#else
          if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
          else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;
#endif
          if (trans) {
            GEMM_OTCOPY(min_l, min_jj, src + (jjs + ls * ld), ld, panel);
          } else {
            GEMM_ONCOPY(min_l, min_jj, src + (ls + jjs * ld), ld, panel);
          }
        }
        cursor += gemm_packed_panel_bytes(pad_l, min_j, GEMM_UNROLL_N);
      }
    }
  }

  if ((size_t)(cursor - data) != expected) return 3;

  memset(&header, 0, sizeof(header));
  header.magic       = GEMM_PACKED_MAGIC;
  header.type_tag    = GEMM_PACKED_TYPE_TAG;
  header.version     = GEMM_PACKED_VERSION;
  header.identifier  = (uint32_t)identifier;
  header.m           = m;
  header.n           = n;
  header.k           = k;
  header.gemm_p      = GEMM_P;
  header.gemm_q      = GEMM_Q;
  header.gemm_r      = GEMM_R;
  header.unroll_m    = GEMM_UNROLL_M;
  header.unroll_n    = GEMM_UNROLL_N;
  header.align_k     = gemm_packed_align_k();
  header.data_offset = (size_t)(data - (char *)dest);
  header.data_size   = (size_t)(cursor - data);
  header.alpha       = (double)alpha;

  memset(dest, 0, GEMM_PACKED_HEADER_BYTES);
  memcpy(dest, &header, sizeof(header));
  return 0;
}

/* Returns 0 when `header`, read from the buffer at `buffer`, describes a packed
 * operand usable for the product described by m, n, k with the expected
 * identifier. */
static int gemm_packed_header_check(const gemm_packed_header_t *header, const void *buffer,
                                    int identifier, BLASLONG m, BLASLONG n, BLASLONG k,
                                    unsigned int type_tag) {
  size_t expected;

  if (header->magic != GEMM_PACKED_MAGIC) return 1;
  if (header->type_tag != type_tag) return 1;
  if (header->version != GEMM_PACKED_VERSION) return 1;
  if (header->identifier != (uint32_t)identifier) return 1;
  if (header->k != k) return 1;
  /* The layout depends on these blocking parameters. They are constant for
   * the life of a process on most targets, but a buffer written by another
   * build or another core selection must be rejected rather than misread. The
   * inner operand does not depend on GEMM_R or GEMM_UNROLL_N, and the outer
   * operand does not depend on GEMM_P, so only what shapes each layout is
   * compared. */
  if (identifier == GEMM_PACKED_IDENTIFIER_A) {
    if (header->m != m) return 1;
    if (header->gemm_p != GEMM_P) return 1;
  } else {
    if (header->n != n) return 1;
    if (header->gemm_r != GEMM_R || header->unroll_n != GEMM_UNROLL_N) return 1;
  }
  if (header->gemm_q != GEMM_Q || header->unroll_m != GEMM_UNROLL_M) return 1;
  if (header->align_k != gemm_packed_align_k()) return 1;
  /* The data offset follows from the buffer address, and the data size from
   * the dimensions and blocking above; a header that disagrees was damaged or
   * belongs to a buffer that was copied to a differently aligned address. */
  if (header->data_offset !=
      (size_t)(gemm_packed_align_ptr((char *)buffer + GEMM_PACKED_HEADER_BYTES) - (char *)buffer)) return 1;
  if (!gemm_packed_data_bytes(identifier, identifier == GEMM_PACKED_IDENTIFIER_A ? m : n, k, &expected)) return 1;
  if (header->data_size != expected) return 1;
  return 0;
}

/*
 * C := beta * C + alpha * op(A) * op(B), where either operand may be a packed
 * buffer produced by ?gemm_packed_pack().
 *
 *   args        m, n, k, a, lda, b, ldb, c, ldc, alpha, beta as for the
 *               regular driver; a and b point at the packed buffers when the
 *               corresponding flag is set, and lda/ldb are then unused
 *   transa/b    0 or 1, only consulted for operands that are not packed
 *   a_packed    non-zero when args->a is a packed buffer
 *   b_packed    non-zero when args->b is a packed buffer
 *   sa, sb      scratch panels laid out as in interface/gemm.c
 *
 * The alpha applied to the product is args->alpha multiplied by the alpha
 * recorded in each packed header.
 *
 * Returns 0 on success. Otherwise bit 0 is set when the packed A buffer is not
 * usable and bit 1 when the packed B buffer is not usable; both headers are
 * checked before returning. Nothing is written to C on failure.
 *
 * When the single precision SBGEMM fallback is active the operands that are
 * not packed are expanded to float and the whole product is handed to
 * sgemm_packed_compute. An expansion that cannot be allocated leaves C
 * untouched and reports success, as an unavailable bfloat16 kernel does.
 */
int GEMM_PACKED_COMPUTE_FN(blas_arg_t *args, int transa, int transb, int a_packed, int b_packed,
                           XFLOAT *sa, XFLOAT *sb, unsigned int type_tag) {
  BLASLONG m = args->m, n = args->n, k = args->k;
  BLASLONG lda = args->lda, ldb = args->ldb, ldc = args->ldc;
  IFLOAT *a = (IFLOAT *)args->a;
  IFLOAT *b = (IFLOAT *)args->b;
  FLOAT *c = (FLOAT *)args->c;
  FLOAT alpha = *(FLOAT *)args->alpha;
  FLOAT beta = *(FLOAT *)args->beta;
  gemm_packed_header_t header_a, header_b;
  const char *a_data = NULL;
  const char *b_data = NULL;
  const char *a_cursor, *b_cursor;
  IFLOAT *sa_cur, *sb_cur;
  BLASLONG ls, is, js, jjs, min_l, min_i, min_j, min_jj, pad_l, l1stride;
  int status = 0;

#ifdef SBGEMM_FLOAT_FALLBACK
  /* The packed buffers were written by sgemm_packed_pack, so their headers
   * are the SGEMM ones and only sgemm_packed_compute may validate them. The
   * scratch was split for float panels by interface/gemm_compute.c. */
  if (sbgemm_float_fallback()) {
    blas_arg_t float_args = *args;
    float *a_float = NULL;
    float *b_float = NULL;
    int fallback_status;
    /* sgemm_packed_compute applies beta and returns without reading either
     * operand once k or alpha is zero, and a packed alpha can only make the
     * product more zero, so the expansion is skipped for those. k itself has
     * to stay as it is, because the header check compares it. */
    int product_is_empty = (k == 0) || (alpha == ZERO);

    if (!a_packed) {
      BLASLONG rows = transa ? k : m;
      BLASLONG cols = product_is_empty ? 0 : (transa ? m : k);
      a_float = sbgemm_expand_to_float(a, rows, cols, lda);
      if (a_float == NULL) {
        openblas_warning(0, SBGEMM_FALLBACK_SKIPPED);
        return 0;
      }
      float_args.a = (void *)a_float;
      float_args.lda = sbgemm_expanded_ld(rows);
    }
    if (!b_packed) {
      BLASLONG rows = transb ? n : k;
      BLASLONG cols = product_is_empty ? 0 : (transb ? k : n);
      b_float = sbgemm_expand_to_float(b, rows, cols, ldb);
      if (b_float == NULL) {
        free(a_float);
        openblas_warning(0, SBGEMM_FALLBACK_SKIPPED);
        return 0;
      }
      float_args.b = (void *)b_float;
      float_args.ldb = sbgemm_expanded_ld(rows);
    }

    fallback_status = sgemm_packed_compute(&float_args, transa, transb, a_packed, b_packed,
                                           (float *)sa, (float *)sb, type_tag);
    free(a_float);
    free(b_float);
    return fallback_status;
  }
#endif

  if (a_packed) {
    memcpy(&header_a, args->a, sizeof(header_a));
    if (gemm_packed_header_check(&header_a, args->a, GEMM_PACKED_IDENTIFIER_A, m, n, k, type_tag)) status |= 1;
  }
  if (b_packed) {
    memcpy(&header_b, args->b, sizeof(header_b));
    if (gemm_packed_header_check(&header_b, args->b, GEMM_PACKED_IDENTIFIER_B, m, n, k, type_tag)) status |= 2;
  }
  if (status) return status;

  if (a_packed) {
    a_data = (const char *)args->a + header_a.data_offset;
    alpha *= (FLOAT)header_a.alpha;
  }
  if (b_packed) {
    b_data = (const char *)args->b + header_b.data_offset;
    alpha *= (FLOAT)header_b.alpha;
  }

  if (m == 0 || n == 0) return 0;

  if (beta != ONE) {
    GEMM_BETA(m, n, 0, beta, NULL, 0, NULL, 0, c, ldc);
  }

  if (k == 0 || alpha == ZERO) return 0;

  b_cursor = b_data;

  for (js = 0; js < n; js += min_j) {
    min_j = gemm_packed_n_block(n - js);
    a_cursor = a_data;

    for (ls = 0; ls < k; ls += min_l) {
      min_l = gemm_packed_k_block(k - ls);
      pad_l = gemm_packed_pad_k(min_l);

      if (b_packed) {
        sb_cur = (IFLOAT *)b_cursor;
        b_cursor += gemm_packed_panel_bytes(pad_l, min_j, GEMM_UNROLL_N);
      } else {
        sb_cur = sb;
      }

      /* First M block. The B panel is copied piecewise and consumed right
       * away, exactly as in level3.c; l1stride == 0 lets the pieces overlap
       * when there is no later M block that would need the whole panel. */
      min_i = m;
      l1stride = 1;
      if (min_i >= GEMM_P * 2) {
        min_i = GEMM_P;
      } else if (min_i > GEMM_P) {
        min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
      } else {
        l1stride = 0;
      }

      if (a_packed) {
        sa_cur = (IFLOAT *)a_cursor;
        a_cursor += gemm_packed_panel_bytes(pad_l, min_i, GEMM_UNROLL_M);
      } else {
        sa_cur = sa;
        if (transa) {
          GEMM_INCOPY(min_l, min_i, a + (ls + 0 * lda), lda, sa_cur);
        } else {
          GEMM_ITCOPY(min_l, min_i, a + (0 + ls * lda), lda, sa_cur);
        }
      }

      if (b_packed) {
        GEMM_KERNEL_N(min_i, min_j, min_l, alpha, sa_cur, sb_cur, c + (0 + js * ldc), ldc);
      } else {
        for (jjs = js; jjs < js + min_j; jjs += min_jj) {
          IFLOAT *panel = sb_cur + pad_l * (jjs - js) * l1stride;
          min_jj = min_j + js - jjs;
#if defined(SKYLAKEX) || defined(COOPERLAKE) || defined(SAPPHIRERAPIDS)
          if (min_jj >= 6 * GEMM_UNROLL_N) min_jj = 6 * GEMM_UNROLL_N;
#else
          if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
          else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;
#endif
          if (transb) {
            GEMM_OTCOPY(min_l, min_jj, b + (jjs + ls * ldb), ldb, panel);
          } else {
            GEMM_ONCOPY(min_l, min_jj, b + (ls + jjs * ldb), ldb, panel);
          }
          GEMM_KERNEL_N(min_i, min_jj, min_l, alpha, sa_cur, panel, c + (0 + jjs * ldc), ldc);
        }
      }

      /* Remaining M blocks reuse the complete B panel. */
      for (is = min_i; is < m; is += min_i) {
        min_i = gemm_packed_m_block(m - is);

        if (a_packed) {
          sa_cur = (IFLOAT *)a_cursor;
          a_cursor += gemm_packed_panel_bytes(pad_l, min_i, GEMM_UNROLL_M);
        } else {
          sa_cur = sa;
          if (transa) {
            GEMM_INCOPY(min_l, min_i, a + (ls + is * lda), lda, sa_cur);
          } else {
            GEMM_ITCOPY(min_l, min_i, a + (is + ls * lda), lda, sa_cur);
          }
        }

        GEMM_KERNEL_N(min_i, min_j, min_l, alpha, sa_cur, sb_cur, c + (is + js * ldc), ldc);
      }
    }
  }

  return 0;
}
