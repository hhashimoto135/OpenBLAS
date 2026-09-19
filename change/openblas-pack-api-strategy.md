# OpenBLAS への MKL 風 packed GEMM API 追加 — 設計と実装

このブランチは上流 OpenBLAS v0.3.34(`e0166008b`)に 1 コミットを重ね、`sgemm` /
`dgemm` / `sbgemm` を MKL と同じ形で `_pack_get_size` / `_pack` / `_compute` に
3 分割した API を追加したものである。本書はその設計と実装をまとめる。
上流との差分パッチは同じディレクトリの `openblas-gemm-pack.patch`(このディレクトリ自身は
含まない)。

## 1. 要約

- 追加する C API(`cblas.h`):
  - `cblas_?gemm_pack_get_size(identifier, m, n, k)` — packed バッファのバイト数
  - `cblas_?gemm_pack(order, identifier, trans, m, n, k, alpha, src, ld, dest)` —
    片方の行列を、カーネルが読むブロック配置へ事前コピー
  - `cblas_?gemm_compute(order, transa, transb, m, n, k, a, lda, b, ldb, beta, c, ldc)` —
    `transa` / `transb` に `CblasPacked` を渡した側を packed バッファとして読む
- コアごとのコードは追加していない。既存のコピー関数(`?gemm_itcopy` 等)と
  microkernel(`?gemm_kernel`)を `gotoblas->` 経由で呼ぶだけなので、`DYNAMIC_ARCH=ON`
  で実行時に選ばれたどのコアでもそのまま動く。
- packed バッファの並びは、通常の GEMM ドライバ `driver/level3/level3.c` が内部スクラッチに
  作る並びと同一。compute は level3.c のループを鏡写しにし、コピー段だけを「packed
  バッファ内のポインタ進行」に置き換えている。カーネルが見るデータは通常の GEMM と
  同一なので、結果はビット単位で `cblas_?gemm` と一致する。
- `alpha` は MKL と同じく `_pack` で受け取る。データは書き換えず、ヘッダに記録して
  compute 時にカーネルへ渡す(§5.6)。
- 呼び出し側から見た安全性: `get_size` は 64 バイト境界のパディングと 16 KiB の末尾余白を
  含み、寸法が大きすぎる場合は 0 を返す。compute は C に触る前に両ヘッダを検証し、
  不正なら xerbla に落とす。

## 2. 変更ファイル

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| 新規 | `driver/level3/gemm_packed.c` | ドライバ本体。`?gemm_packed_size` / `?gemm_packed_pack` / `?gemm_packed_compute`。型ごとに 3 回コンパイル(`sgemm_packed` / `dgemm_packed` / `sbgemm_packed`) |
| 新規 | `interface/gemm_pack.c` | `cblas_?gemm_pack` と `cblas_?gemm_pack_get_size`(CNAME から名前を派生) |
| 新規 | `interface/gemm_compute.c` | `cblas_?gemm_compute` |
| 新規 | `interface/gemm_packed_common.h` | 2 つの interface ファイルが共有する型ディスパッチ、trans 変換、AMX 権限取得 |
| 変更 | `cblas.h` | `CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162}`、`CBLAS_STORAGE {CblasPacked=151}`、9 関数のプロトタイプ |
| 変更 | `common_level3.h` | ドライバ関数のプロトタイプ、`GEMM_PACKED_*` 定数 |
| 変更 | `interface/CMakeLists.txt`, `driver/level3/CMakeLists.txt` | `GenerateNamedObjects` でオブジェクト生成 |
| 変更 | `interface/Makefile`, `driver/level3/Makefile` | Makefile ビルド向けオブジェクト一覧とルール(SUFFIX / PSUFFIX) |
| 変更 | `exports/gensymbol`, `exports/gensymbol.pl` | 共有ライブラリのエクスポート一覧に 9 シンボル追加 |
| 新規 | `utest/test_extensions/test_{s,d,sb}gemm_pack.c` | 同梱ユニットテスト(s 36 件、d 36 件、sb 16 件) |
| 変更 | `utest/CMakeLists.txt`, `utest/Makefile` | 上記テストの登録 |
| 新規 | `change/` | 本書と差分パッチ(ビルドには関与しない) |

## 3. API 仕様

```c
typedef enum CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162} CBLAS_IDENTIFIER;
typedef enum CBLAS_STORAGE    {CblasPacked=151} CBLAS_STORAGE;

size_t cblas_sgemm_pack_get_size(CBLAS_IDENTIFIER identifier, blasint m, blasint n, blasint k);
void   cblas_sgemm_pack(CBLAS_ORDER order, CBLAS_IDENTIFIER identifier, CBLAS_TRANSPOSE trans,
                        blasint m, blasint n, blasint k, float alpha,
                        const float *src, blasint ld, float *dest);
void   cblas_sgemm_compute(CBLAS_ORDER order, blasint transa, blasint transb,
                           blasint m, blasint n, blasint k,
                           const float *a, blasint lda, const float *b, blasint ldb,
                           float beta, float *c, blasint ldc);
/* dgemm は double。sbgemm は src / a / b が bfloat16、alpha / beta / c が float */
```

`m, n, k` は常に積 `C(m×n) = op(A)(m×k) · op(B)(k×n)` の寸法。`identifier` は pack する
行列が A か B か、`trans` はその行列に適用する転置。`_compute` の `transa` / `transb` は
`CblasNoTrans` / `CblasTrans` / `CblasPacked` のいずれか。packed 側の `ld` は無視する。

### 3.1 MKL との対応

| 項目 | MKL | 本実装 |
| --- | --- | --- |
| 関数名・引数順 | `cblas_?gemm_pack_get_size` / `_pack` / `_compute` | 同じ(`MKL_INT` → `blasint`) |
| `alpha` | `_pack` で受け取り、`_compute` には無い | 同じ。ヘッダに記録し、compute 時にカーネルへ渡す(データは書き換えない) |
| 両オペランド packed 時の alpha | 文書上は未規定 | 2 つの記録値の積 |
| packed 側の `ld` | 無視 | 無視(検証もしない) |
| `m, n, k` の再利用 | pack と compute で同じ値を要求 | `k` と packed 側の extent(A なら m、B なら n)は検証。もう一方の extent は記録のみで、異なる値で compute してよい |
| 対応型 | s, d, c, z(bf16 は別 API) | s, d, sb |
| バッファ確保 | 呼び出し側 | 呼び出し側。`get_size` は order 非依存で、row-major / column-major どちらの配置にも足りる大きさを返す |
| `dest` のアライメント | 要素型 | 要素型(ヘッダは `memcpy` で出し入れするため 8 バイト境界は不要) |

### 3.2 エラー報告

既存 GEMM と同じく `xerbla` に渡す。ルーチン名は `"SGEMM_PACK_GET_SIZE "`, `"SGEMM_PACK "`,
`"SGEMM_COMPUTE "`(d / sb も同様)。`info` は引数位置で、複数の違反があれば最小の
番号を報告する。

| 関数 | info | 意味 |
| --- | --- | --- |
| `_pack_get_size` | 1 / 2 / 3 / 4 | identifier / m / n / k が不正(負の寸法を含む) |
| `_pack_get_size` | 0 | packed サイズが `PTRDIFF_MAX` を超える(戻り値 0) |
| `_pack` | 1 〜 6 | order / identifier / trans / m / n / k |
| `_pack` | 8 / 9 / 10 | src が NULL(非空のとき)/ ld が格納行数未満 / dest が NULL |
| `_pack` | 0 | サイズ超過(`_pack_get_size` が 0 を返す寸法)、または内部整合性エラー(§5.3) |
| `_compute` | 1 〜 6 | order / transa / transb / m / n / k |
| `_compute` | 7 / 9 | a / b が NULL(packed のとき、または非空で読まれるとき)、またはヘッダ検証失敗(§5.5)。両方不正なら 7 |
| `_compute` | 8 / 10 / 13 | lda / ldb(packed でない側のみ)/ ldc |

`info` はすべてユーザの引数位置で報告する。row-major では内部で A と B が入れ替わるが、
`interface/gemm_compute.c` が `ld` 検査とヘッダ検証の結果をユーザ側の位置(a = 7 / 8、
b = 9 / 10)へ写像し直す。

## 4. 前提となる OpenBLAS の内部構造

### 4.1 通常の GEMM の流れ

1. `interface/gemm.c`(`cblas_sgemm` 等)が引数を検査し、`CblasRowMajor` なら
   `C^T = B^T A^T` として **A と B を入れ替え、m と n を入れ替え**、内部は常に
   column-major で扱う。
2. `blas_memory_alloc(0)` で `BUFFER_SIZE` のスクラッチを 1 本借り、前半 `sa`
   (`GEMM_P × GEMM_Q` 要素)と後半 `sb`(`GEMM_Q × GEMM_R` 要素)に分ける。
3. `driver/level3/level3.c`(`sgemm_nn` 等、NN/NT/TN/TT で 4 通り)が 3 重ブロック:
   - `js`: N 方向を `GEMM_R` 幅
   - `ls`: K 方向を `GEMM_Q` 幅。残りが `GEMM_Q` 超 `2*GEMM_Q` 未満なら半分に割り、
     `GEMM_UNROLL_M` の倍数へ丸める
   - `is`: M 方向を `GEMM_P` 幅。同様の半分割規則
   - 各 (ls, is) で A の小片を `GEMM_ITCOPY` / `GEMM_INCOPY` で `sa` へ、各 (ls, js) で
     B の小片を `GEMM_ONCOPY` / `GEMM_OTCOPY` で `sb` へ **`min_jj` 列ずつ**コピーし
     (`sb + pad_min_l * (jjs - js) * l1stride` にオフセット)、`GEMM_KERNEL` を呼ぶ。
   - 最初の M ブロックだけは B のコピーとカーネルを `min_jj` 単位で交互に行い、
     2 つ目以降の M ブロックは `sb` 全体を 1 回のカーネル呼び出しで消費する。
     M ブロックが 1 つしかない(`m <= GEMM_P`)場合は `l1stride = 0` として小片を
     同じ位置へ上書きする(L1 に収める最適化)。
4. `beta != 1` なら先頭で `GEMM_BETA` が C 全体をスケールし、`k == 0` や
   `alpha == 0` はそこで終わる。alpha はカーネルが累積和に掛ける。

### 4.2 DYNAMIC_ARCH

- `kernel/` だけがコアごとにコンパイルされ、関数ポインタとブロッキング定数を
  `gotoblas_t` テーブル(`common_param.h`)へ登録する(`kernel/setparam-ref.c`)。
- `driver/` と `interface/` は 1 回だけコンパイルされ、`GEMM_P` は `gotoblas->sgemm_p`、
  `GEMM_KERNEL_N` は `gotoblas->sgemm_kernel` のように **すべてテーブル経由**になる
  (`common_param.h`, `common_s.h` 等の `#ifdef DYNAMIC_ARCH` 節)。
- したがって、新しいドライバを driver/level3 に置き、既存マクロだけで書けば、
  コアごとのコード追加もテーブル拡張も不要で DYNAMIC_ARCH に対応できる。
  本実装はこの方針を採り、`gotoblas_t` には手を入れていない。

## 5. 実装

### 5.1 基本方針

「pack = level3.c がスクラッチに作るはずの並びを、行列全体分まとめて別バッファに作る。
compute = level3.c と同じループを回し、コピー段を packed バッファ内のポインタ進行に
置き換える」。

ブロック境界を決める規則は `gemm_packed_k_block()` / `gemm_packed_m_block()` /
`gemm_packed_n_block()` / `gemm_packed_pad_k()` / `gemm_packed_panel_bytes()` の
inline 関数に集約し、size / pack / compute の 3 者が同じ関数を同じ順序で使うことで
オフセットの一致を保証している。

### 5.2 packed バッファの配置

```
+-----------------------------+  dest(ユーザ確保、要素型のアラインで十分)
| gemm_packed_header_t        |  GEMM_PACKED_HEADER_BYTES (128) の予約領域に memcpy で
|                             |  出し入れ: magic, type_tag('s'/'d'/'sb'), version,
|                             |  identifier, m, n, k, gemm_p/q/r, unroll_m/n, align_k,
|                             |  data_offset, data_size, alpha (double)
+-----------------------------+
| (パディング)                 |  data を 64 バイト境界へ(最大 63 バイト)
+-----------------------------+  data = dest + data_offset
| panel[0]                    |  各 panel は 64 バイト境界から始まる
| panel[1]                    |
| ...                         |
+-----------------------------+  data + data_size
| 末尾余白 GEMM_PACKED_TAIL_GUARD (16 KiB) |
+-----------------------------+  dest + get_size
```

- ヘッダを `memcpy` で扱うのは、`dest` が `float *` / `bfloat16 *` として渡されるため
  8 バイト境界にある保証がなく、構造体ポインタで直接触ると C 規格上 UB になるため。
  `sizeof(gemm_packed_header_t) <= GEMM_PACKED_HEADER_BYTES` はコンパイル時に検査する。
- **内部 A(inner オペランド)**: `for ls: for is:` の順に panel (ls, is) を並べる。
  panel の中身は `GEMM_ITCOPY(min_l, min_i, ...)`(no-trans)/ `GEMM_INCOPY`(trans)
  の出力そのもの。
- **内部 B(outer オペランド)**: `for js: for ls:` の順に panel (js, ls) を並べる。
  panel の中身は level3.c と同じ `jjs` 分割で `GEMM_ONCOPY` / `GEMM_OTCOPY` を呼び、
  `l1stride = 1` 相当のオフセット `pad_min_l * (jjs - js)` に書く。
- panel 予約サイズ = `pad_k × round_up(extent, UNROLL) × sizeof(elem)` を 64 バイトへ
  丸めたもの。`extent` の切り上げは、末尾をレジスタブロック幅までパディングする
  コピーカーネルが現れても書き過ぎないための余裕(x86 の既存カーネルは
  ちょうど `m × n` 要素を書く)。
- **末尾余白 `GEMM_PACKED_TAIL_GUARD`(16 KiB)**: microkernel は「パネルは巨大な
  スクラッチの中にある」前提で書かれており、K 方向のソフトウェアパイプラインで最終
  パネルの終端を越えて読む(ガードページ試験の実測: Cooperlake / SkylakeX / SandyBridge /
  Nehalem は 256 B 未満、Prescott は 512 B 未満、Core2 は 2 KiB 未満、Haswell / Zen は 0)。
  `get_size` はこの余白を加算し、バッファ末尾がページ境界に接していても読み越しが
  SIGSEGV にならないようにする。余白は読まれるだけで書かれない。
- `pad_k` は bf16 の `SBGEMM_ALIGN_K`(`gotoblas->sbgemm_align_k`)で K を切り上げる。
  x86 では 1(パディングなし)。level3.c が `sb` のオフセット計算に使う
  `pad_min_l` と同じ値。
- `get_size` は identifier と `m, n, k` だけを受け取り order を知らないので、
  inner 配置と outer 配置の両方を計算して大きい方を返す。両者の差は panel 単位の
  64 バイト丸めと `UNROLL` 切り上げ分だけで、本体はどちらも `≈ pad(k) × extent` 要素。

### 5.3 サイズ計算(閉形式・オーバーフロー検査)

K / M のブロック列は「`block` が `full_count` 回、そのあと末尾ブロックが 1〜2 個」と
決まっている(`gemm_packed_split()`: 残りが `2*block` 以上なら `full_count = 残り/block - 1`、
残りが `block` を超えれば半分割して `UNROLL` へ丸め、さもなくば残り全部を 1 ブロック)。
N 方向は `GEMM_R` の等分割と余り 1 ブロック。したがってバイト数はループで列挙せず
O(1) で求まる(`gemm_packed_data_bytes()`)。

`size_t` の加算・乗算は `gemm_packed_add()` / `gemm_packed_mul()` で `PTRDIFF_MAX` を上限に
検査し、超えれば `get_size` は 0 を返し(xerbla info 0)、`pack` も同じ寸法を info 0 で
拒否する。`INT_MAX` 級の寸法でも即時に返る。

pack はさらに、書き終えたカーソル位置が閉形式の `data_size` と一致することを検証し、
不一致なら内部エラー(xerbla info 0)としてヘッダを書かず、後続の compute が確定的に
拒否するようにしている。size / pack / compute の 3 者の食い違いはここで顕在化する。

### 5.4 compute

4 通り(A のみ packed / B のみ / 両方 / どちらも packed でない)を 1 つのループで扱う。

```
for js (GEMM_R 幅):
  a_cursor = packed A の先頭            # A は js ごとに最初から読み直す
  for ls (GEMM_Q 幅, level3.c と同じ半分割規則):
    sb_cur = packed B なら b_cursor(連続に進む) / さもなくば scratch sb
    最初の M ブロック:
      sa_cur = packed A なら a_cursor / さもなくば ITCOPY|INCOPY → scratch sa
      B が packed        : KERNEL(min_i, min_j 全体)
      B が packed でない : level3.c と同じ jjs 分割で OCOPY と KERNEL を交互に
                           (l1stride は level3.c と同じ規則)
    2 つ目以降の M ブロック:
      sa_cur を進める or コピーし、KERNEL(min_i, min_j 全体, sb_cur)
```

- `m == 0 || n == 0` は gemm.c と同じく beta 適用前に return。`beta != 1` なら
  `GEMM_BETA`、その後 `k == 0` / `alpha == 0` で return(level3.c と同順)。
- スクラッチは gemm.c と同じ `blas_memory_alloc(0)` から借り、`sa` / `sb` の切り方も
  gemm.c と同じ。packed でない側だけがスクラッチを使う。
- packed B を「min_j 全体で 1 回」消費するのは、level3.c が 2 つ目以降の M ブロックで
  同じことをしている(`jjs` 分割で書いた `sb` を全体で読む)ことに依拠している。
  分割境界は `UNROLL_N` の倍数(末尾を除く)なので、コピーの出力並びは分割の有無で
  変わらない。
- `SKYLAKEX` / `COOPERLAKE` / `SAPPHIRERAPIDS` で level3.c が使う `min_jj` の丸め規則も
  そのまま写している。

### 5.5 ヘッダ検証

compute は C に触る前に両方の packed ヘッダを `memcpy` で取り出して検査する:

1. magic、型タグ(`sgemm_pack` のバッファを `sbgemm_compute` に渡した等を検出)、version
2. identifier(A / B の取り違え)
3. `k`、自分の extent(A なら m、B なら n)
4. その配置が依存するブロッキング定数が現在の `gotoblas` テーブルと一致すること
   (A: `GEMM_P`, `GEMM_Q`, `UNROLL_M`, `align_k` / B: `GEMM_R`, `GEMM_Q`, `UNROLL_M`,
   `UNROLL_N`, `align_k`)。x86 ではプロセス内で定数だが、別ビルドや別コアで作られた
   バッファ、ファイルから読んだバッファを誤読しないための保険になる
5. `data_offset` をバッファの実アドレスから再計算した値と一致すること(別アラインメントの
   アドレスへ `memcpy` されたバッファを検出)
6. `data_size` を閉形式で再計算した値と一致すること(改竄・破損の検出)

失敗時はビットマスク(bit0 = A、bit1 = B)を interface へ返し、interface がユーザの
引数位置(a = 7、b = 9)へ写像して小さい方を xerbla に報告する。C は変更されない。

### 5.6 alpha の扱い

- `_pack` の `alpha` はヘッダに `double` で記録するだけで、データには掛けない。
  bf16 でも丸めが増えず、`alpha` を変えたいときに再パックが要らない。
- compute の有効 alpha = `args->alpha`(interface は 1.0 を渡す)× 各 packed ヘッダの alpha。
  これをカーネルの alpha としてそのまま渡すので、加算順序と丸めは `cblas_?gemm` と
  同一になる(alpha が NaN / 0 / ∞ の場合の結果も一致する)。
- alpha の有限性や積のオーバーフローは MKL と同様に C 層では検査しない。必要なら
  呼び出し側で行う。

### 5.7 bf16(sbgemm)固有の注意

- `SAPPHIRERAPIDS` の ONCOPY / OTCOPY(`sbgemm_o?copy_16_spr.c`)は AMX タイル命令を
  使うため、pack 側でも gemm.c と同じ `arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA)`
  を実行する(`gemm_packed_kernels_ready()`)。DYNAMIC_ARCH では
  `gotoblas->need_amxtile_permission` を見る。Linux のこの権限はプロセス単位で、fork で
  継承され exec で消える。
- 権限が得られない場合は上流 `cblas_sbgemm` と同じくメッセージを出して何もしないが、
  pack はヘッダ領域をゼロ化して後続の compute が確定的に拒否するようにしている。
- 取得済みフラグ(`gemm_packed_amxtile_permission`)の読み書きは `__atomic_load_n` /
  `__atomic_store_n`(relaxed)で行い、初回呼び出しの競合は最悪でも syscall の重複に
  留まる。
- `COOPERLAKE` / `SPR` のコピーカーネルは奇数 K の末尾行を単独で書き、要素数は
  `k × n` ちょうど。`ALIGN_K` が 1 でないアーキテクチャでは `pad_k` 分を確保する。

### 5.8 スレッド安全性

- compute のスクラッチは gemm.c と同じ `blas_memory_alloc(0)`(`USE_LOCKING` で
  排他)から借りるので、複数スレッドが **同じ packed バッファを読み取り専用で共有**
  しながら compute を呼べる。
- packed バッファへの書き込み(pack)と読み取り(compute)の同時実行は呼び出し側の
  責任。
- compute 自体は常にシングルスレッドで走る(`USE_THREAD=ON` ビルドでもスレッド分割
  しない)。

## 6. ビルドシステム統合

- CMake: `GenerateNamedObjects`(`cmake/utils.cmake`)を既存 gemm と同じ引数パターンで
  呼ぶ。interface は `CBLAS_FLAG == 1` のループ内で `gemm_pack.c; gemm_compute.c` を
  real 型のみ(第 8 引数 `1`)、bf16 は `"sbgemm_pack"` / `"sbgemm_compute"` 名で生成。
  driver/level3 は `gemm_packed.c` を real 型 + `BFLOAT16` で生成し、`CNAME` から
  `_size` / `_pack` / `_compute` を派生させる。
- オブジェクト名の衝突回避: interface 側は `cblas_?gemm_pack.o` / `cblas_?gemm_compute.o`、
  driver 側は `?gemm_packed.o`。Fortran 風の `sgemm_pack.o` を将来 interface に
  追加しても driver と同名にならない(`ar` は同名メンバを置換するため重要)。
- Makefile: `Makefile.system` が対象名の stem から `-DCNAME=` を生成する仕組みに乗せ、
  `CSBLAS3OBJS` / `CDBLAS3OBJS` / `CSBBLAS3OBJS` と `SBLASOBJS` / `DBLASOBJS` /
  `SBBLASOBJS` にオブジェクトを追加、`$(SUFFIX)` と `$(PSUFFIX)` の両ルールを追加した。
- `exports/gensymbol`(sh)と `gensymbol.pl` の `cblasobjss` / `cblasobjsd` /
  `bfcblasobjs` へシンボル追加。静的ライブラリには無関係だが、共有ライブラリと
  Windows の `.def` 生成で必要。
- utest: `test_{s,d,sb}gemm_pack.c` の本体は `#ifndef NO_CBLAS` で囲み(既存
  `test_sgemmt.c` と同じ流儀)、`make NO_CBLAS=1` でもリンクできる。

## 7. 制限事項・非対応

- 複素数(`cgemm` / `zgemm`)、`bgemm`(bf16 出力)、`shgemm`(fp16)は対象外
  (`#error` で弾く)。
- compute はシングルスレッド。マルチスレッド化するなら level3_thread.c 相当の
  分割を packed 配置に合わせて書く必要がある。
- gemm.c の small-matrix kernel、`n == 1` / `m == 1` の gemv 転送、`sgemm_direct` は
  compute では使わない(事前 pack 済みデータはブロック配置なので)。
- Fortran 風の `sgemm_pack_` 等は提供しない(MKL の C API のみ)。
- LoongArch / MIPS64 の SMP 非 DYNAMIC_ARCH ビルドでは `openblas_set_num_threads()` が
  `blas_set_parameter()` を再実行して `gemm_q` / `gemm_r` を変えることがある。packed
  バッファは作成時のスレッド設定でのみ有効(ヘッダ検証で拒否される)。
- `other_extent`(MKL の第 3 次元)は記録するだけで検証しない。厳密に MKL と同じ
  契約にしたい場合は `gemm_packed_header_check()` に `header->n == n`(A の場合)/
  `header->m == m`(B の場合)を追加すればよい。
- Windows / MSVC ビルド、`INTERFACE64`、`USE_OPENMP` は未検証(コードは
  `blasint` / `BLASLONG` を通しており、既存 gemm.c と同じ前提で書いている)。

## 8. 検証

環境: AMD Ryzen 7 9800X3D(AVX-512 / avx512_bf16 → DYNAMIC_ARCH は Zen 用コア)、
Linux(WSL2)、GCC 16.2、CMake 4.4、Ninja。他コアは `OPENBLAS_CORETYPE` で強制。

| テスト | 内容 | 結果 |
| --- | --- | --- |
| C スモーク(float / double) | 13 形状 × RowMajor/ColMajor × 4 trans × {A packed, B packed, 両方, なし} = 832 ケース、`cblas_?gemm` とビット一致を確認、番兵でバッファ超過検出 | 0 failures |
| C スモーク(bf16) | 15 形状 × 同上 = 480 ケース、`cblas_sbgemm` と比較 | 0 failures |
| ガードページ試験 | packed バッファの直後を `PROT_NONE` ページにし(余白 0)、26 寸法の組合せ × 全 order / trans / packed 組合せ × s, d, sb = 215,424 ケースを pack + compute。`get_size` の範囲外を読めば SIGSEGV | Zen(自動選択)、SkylakeX、Haswell、SandyBridge、Nehalem で読み越し 0、結果不一致 0 |
| OpenBLAS `openblas_utest_ext`(スタンドアロン CMake、`USE_THREAD=ON`, `DYNAMIC_ARCH=ON`) | 既存 607 件 + 新規 88 件(xerbla の info 番号、ヘッダ不一致・identifier 取り違え・extent 不一致・`data_offset` ずれ・ゼロ埋め・両方不正・他型バッファの拒否と C 不変、NULL 引数、サイズ超過、全末尾長での `get_size` と実配置の一致、ガードページ、m 変更再利用、要素アラインのみのバッファ、`GEMM_R` 跨ぎ) | 695 / 695 OK(`OPENBLAS_NUM_THREADS=1` でも同じ) |
| Makefile ビルド(純正ツリー + パッチ、`DYNAMIC_ARCH=1`、複素数型込み、`USE_THREAD=0`) | ビルド成功、18 シンボル存在、`make -C utest` の utest / utest_ext、上記 C スモーク | 全件 OK |
| 利用側(alinalg)gtest | Release 460 件(新規 17 件)、Debug 487 件(新規 15 件 + death test 4 件)。`alin::dot` との厳密一致 | 全件合格 |
| 警告 | `-Wall -Wextra -Wconversion` で新規ソースを個別コンパイル | 0 |

形状は `k = 1000`(`GEMM_Q` を超え、末尾の半分割経路を通る)、`m, n = 1700`
(`GEMM_P` を跨ぐ)、`m = 70000 / 40000 / 140000`(s / d / sb で `GEMM_R` を跨ぐ。
DYNAMIC_ARCH の `gemm_r` は BUFFER_SIZE 由来で本ビルドでは 32,240〜130,800)、
`k = 1, 3`(bf16 の奇数 K)、`m/n/k = 0`、`INT_MAX` 級の寸法(`get_size` が 0 を返す)を含む。

ベンチマーク(Release, `-O3 -march=native`, 単一スレッド。`gemm` は毎回両オペランドを
コピー、`compute` は右オペランドを事前 pack):

| type | m × k × n | gemm [ms] | compute [ms] | 倍率 | pack [ms] |
| --- | --- | --- | --- | --- | --- |
| float | 1 × 512 × 2048 | 0.204 | 0.051 | 4.0× | 0.198 |
| float | 16 × 512 × 2048 | 0.316 | 0.118 | 2.7× | 0.233 |
| float | 64 × 1024 × 4096 | 2.53 | 1.70 | 1.5× | 0.424 |
| float | 256 × 1024 × 4096 | 7.74 | 6.92 | 1.1× | 0.475 |
| double | 16 × 512 × 2048 | 0.412 | 0.283 | 1.5× | 0.149 |
| double | 64 × 1024 × 4096 | 4.75 | 4.21 | 1.1× | 0.829 |
| bfloat16 | 16 × 512 × 2048 | 0.098 | 0.078 | 1.3× | 0.045 |
| bfloat16 | 64 × 1024 × 4096 | 1.31 | 0.95 | 1.4× | 0.178 |

小さい m(推論のバッチ 1〜16)で効果が大きく、m が大きくなるとコピーコストの
比率が下がるので差は縮む。

## 9. 既知の上流の問題(本パッチでは修正していない)

- ~~`kernel/setparam-ref.c` の Zen 向けブロッキング調整は、`OPENBLAS_CORETYPE` で別コアを
  強制した場合にもそのコアのテーブルへ適用される。Zen ホストで `PRESCOTT` / `CORE2` を
  強制すると、packed API と無関係な `cblas_sgemm`(例: m=1, n=5, k=300)がプロセスの
  配置次第で `sgemm_kernel` 内の SIGSEGV を起こす。このためガードページ試験はこの 2 コアでは
  完走できない(旧来の実測で読み越しは 512 B / 2 KiB 未満と分かっており、16 KiB の余白で
  覆われる)。~~
  → このブランチの 3 つ目のコミットで修正した。`change/openblas-zen-blocking-override.md`
  を参照。SIGSEGV は読み越しではなくカーネル内のスタックバッファ溢れで、`sgemm` だけでなく
  `dgemm` も、packed API の有無と無関係に起きていた。ガードページ試験も全コアで完走する。
- `kernel/x86_64/tobf16.c` の SMP 経路は要素数が 100,000 を超えると `nthreads` を最低 4 に
  引き上げるため、`USE_THREAD=ON` ビルドで `OPENBLAS_NUM_THREADS=1` にすると
  `cblas_sbstobf16` が停止することがある。utest ではビットシフトで bf16 に変換して回避。

## 10. 利用側(alinalg)での使い方の要点

alinalg は本 API を `alin::dot_prepare_pack` / `alin::dot_compute` として公開する。行列は
row-major なので `CblasRowMajor` で呼び、ユーザの右オペランド(`kRight`)が OpenBLAS 内部の
inner オペランド(`GEMM_P` でブロック)、左オペランド(`kLeft`)が outer オペランドになる。
この写像は interface 層が行うので利用側は identifier を渡すだけでよい。

- `get_size` → 確保 → `pack` の順で `DotPacked<T>` を用意し、`compute` は
  `CblasPacked` を該当側に渡す。`other_extent`(MKL の第 3 次元)は配置に影響しないので
  1 つの packed 重み行列を任意のバッチサイズに使える。
- alpha の有限性や記録された 2 つの alpha の積のオーバーフローなど、C 層で検査しない
  条件は利用側の assert で扱う。
- AMX 権限が得られない環境では pack が黙って何もしないため(§5.7)、利用側は成功を
  検出できない。この挙動は利用側のドキュメントに明記する。

## 11. 今後の拡張候補

- 複素数対応: `KERNEL_FUNC` を conj の組合せ(N/L/R/B)で選ぶ分岐と、alpha を 2 要素で
  扱う変更が必要。ドライバ構造はそのまま使える。
- compute のマルチスレッド化(`gemm_thread_mn` 相当を packed 配置に合わせて分割)。
- MKL の bf16 API(`cblas_gemm_bf16bf16f32_pack` 系)に合わせた別名の提供。
