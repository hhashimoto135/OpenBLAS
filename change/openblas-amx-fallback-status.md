# AMX が使えないときの Cooperlake カーネルへのフォールバックと、bfloat16 GEMM のステータス返却

このブランチ(`alinalg-pack-api`)の 4 つ目のコミット。次の 2 つを追加する。

1. Sapphire Rapids で AMX が使えないとき、計算をスキップする代わりに **AVX512-BF16 の
   Cooperlake の bfloat16 GEMM カーネルで計算する**(`DYNAMIC_ARCH`)。Cooperlake テーブルの
   無いビルドでは 2 つ目のコミットの単精度フォールバック(`SGEMM`、AVX512)が引き受ける。
2. bfloat16 GEMM の失敗、とりわけ単精度フォールバックの一時領域の **確保失敗を
   呼び出し側へ戻り値で伝える**。`cblas_?gemm_pack` / `cblas_?gemm_compute` は `int`
   を返し、`cblas_sbgemm` には同じ本体で値を返す `cblas_sbgemm_status` を添える。

上流との差分パッチは同じディレクトリの `openblas-amx-fallback-status.patch`。
1〜3 つ目のコミットの設計は `openblas-pack-api-strategy.md`、
`openblas-sbgemm-float-fallback.md`、`openblas-zen-blocking-override.md`。

## 1. 背景

### 1.1 AMX

Sapphire Rapids の bfloat16 GEMM カーネル(`sbgemm_kernel_16x16_spr.c`、
`sbgemm_o?copy_16_spr.c`)は AMX タイル命令を使う。AMX は

- CPU が持ち(CPUID.7.0:EDX の AMX-TILE / AMX-BF16)、
- OS が有効にし(XCR0 のビット 17, 18)、
- Linux ではさらにプロセスごとに `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`
  で許可を得る

必要がある。5.16 より古いカーネルや、この syscall を遮断するコンテナでは許可が下りない。
上流は最初の `SBGEMM` 呼び出しで許可を求め、拒否されると
「XTILEDATA permission not granted ... skip sbgemm calculation」を出して **何も計算せずに
戻る**。呼び出し側からはそれを検出できない(`cblas_sbgemm` は void)。

Sapphire Rapids は AVX512-BF16 も持つので、Cooperlake の bfloat16 GEMM カーネル
(`KERNEL.COOPERLAKE` の `SBGEMM*`)はそのまま動く。`KERNEL.SAPPHIRERAPIDS` は
`KERNEL.COOPERLAKE` を include して `SBGEMM*` だけを上書きしているので、カーネルの
違いは bfloat16 GEMM に限られる。**ブロッキングパラメータは違う**: `param.h` の
`COOPERLAKE` 節には `L3_SIZE % 32 MiB == 0 && L2_SIZE == 1 MiB` のとき `SGEMM` /
`DGEMM` / `CGEMM` / `ZGEMM` の P/Q を変える分岐があり、`SAPPHIRERAPIDS` 節には無い
(§2.1 でテーブルごと差し替えない理由)。

### 1.2 確保失敗

2 つ目のコミットの単精度フォールバックは、オペランドの float 展開を `malloc` で確保する。
失敗時は `C`(pack ならヘッダ)に触れず `openblas_warning` を出して戻るだけで、
`cblas_sbgemm` と `cblas_sbgemm_compute` は void、`cblas_sbgemm_pack` は `xerbla(info=0)`
であり、いずれも呼び出し側(alinalg)は失敗を知れなかった。

## 2. AMX フォールバック

### 2.1 判定と置き換え(`driver/others/sbgemm_amx.c`、新規)

プロセスで一度だけ解決し、結果をキャッシュする。

```
sbgemm_kernels_unavailable():
  state が未解決なら blas_lock で 1 スレッドだけが解決する
    DYNAMIC_ARCH:
      gotoblas->need_amxtile_permission == 0  → READY(Sapphire Rapids 以外)
      AMX が使える(CPUID/XCR0、Linux なら arch_prctl も成功) → READY
      gotoblas_sbgemm_use_cooperlake() == 0    → READY(bfloat16 GEMM エントリを Cooperlake のものに)
      さもなくば                               → UNAVAILABLE
    静的 SAPPHIRERAPIDS ビルド:
      AMX が使える → READY、さもなくば UNAVAILABLE
    それ以外のビルド: READY
  戻り値: state == UNAVAILABLE
```

- 許可要求は **遅延のまま**(最初の bfloat16 GEMM で行う)。上流と同じで、bfloat16 を
  使わないプロセスに AMX 許可の副作用(カーネルが許可済み機能ぶんの signal frame を
  要求するようになる)を持ち込まない。
- `gotoblas_sbgemm_use_cooperlake()`(`driver/others/dynamic.c`)は、Cooperlake テーブルの
  `init()` を済ませてから、**いま使っているテーブルの SBGEMM フィールドだけ**を Cooperlake の
  値で上書きする: `sbgemm_p/q/r`、`sbgemm_unroll_m/n/mn`、`sbgemm_align_k`、
  `sbgemm_kernel`、`sbgemm_beta`、`sbgemm_{in,it,on,ot}copy`、`SMALL_MATRIX_OPT` の
  `sbgemm_small_*`、そして `need_amxtile_permission = 0`。`gotoblas` ポインタは動かさない。
  - **テーブルごと差し替えない理由**: §1.1 のとおり 2 つの `param.h` 節は `SGEMM` 等の P/Q
    でも違い得る。ポインタを差し替えると、他スレッドで実行中の `sgemm` が `level3.c` の
    ブロックごとに読み直す `GEMM_P` / `GEMM_Q` が途中で変わり、呼び出し時に切ったスクラッチ
    `sa` を溢れ得る(このホストでは Zen 4/5 の実行時上書きが両テーブルを同じ値に揃えるので
    テストでは見えないが、Intel の Sapphire Rapids ホストでは揃わない)。
  - **SBGEMM フィールドだけなら安全な理由**: それらを読むのは bfloat16 GEMM のエントリ
    ポイントだけで、それらは §2.3 のとおり必ず先にこの関数を通る。解決は `blas_lock` 内で
    行い、書き終えてから `MB` を挟んで state を公開するので、READY を見たスレッドは
    書き換え後の値を読む。
- `sbgemm_float_fallback()`(`common_sbfallback.h`)は先頭でこの関数を呼び、
  `DYNAMIC_ARCH` では `gotoblas_corename()` を見る。名前は `"SapphireRapids"` のままだが、
  bfloat16 カーネルは Cooperlake のものになっているので「bfloat16 カーネルあり」の判定で
  正しい。静的 `SAPPHIRERAPIDS` ビルドでは UNAVAILABLE をそのまま単精度フォールバックの
  条件にする。
- `DYNAMIC_LIST` ビルドで `DYN_COOPERLAKE` が無い場合、Cooperlake テーブルは別コアの別名
  なので `gotoblas_sbgemm_use_cooperlake()` は失敗を返し、単精度フォールバックに任せる。
  `NO_AVX512` ビルドでは `gotoblas_SAPPHIRERAPIDS` 自体が別名で `need_amxtile_permission`
  が立たないので、そもそもここへ来ない。

### 2.2 起動時の置き換え(`driver/others/dynamic.c`)

`get_coretype()` は Sapphire Rapids を選ぶ前に `support_amx_bf16()` を見るが、
`force_coretype()` は見ない。`gotoblas_dynamic_init()` の末尾で、選ばれたテーブルが
AMX を要し `support_amx_bf16()` が偽なら、その場で同じ置き換えを行う
(`FALLBACK_VERBOSE` で `AMX_FALLBACK` を出す)。強制された Sapphire Rapids が AMX の無い
ホストや OS で AMX カーネルを実行することはなくなる。Linux のプロセス許可は §2.1 の
遅延判定に残す。`openblas_get_corename()` は `"SapphireRapids"` を返し続ける。

あわせて上流の 2 点を直した。

- `force_coretype()` のループ上限が `i <= 25` で、`corename[26]` の `"SapphireRapids"`
  が認識されなかった(3 つ目のコミットの設計書 §4.1 で指摘済み)。`26` にし、
  `case 26: return &gotoblas_SAPPHIRERAPIDS` を追加した。
- `support_amx_bf16()` は「AMX enabled」のつもりで CPUID.D.0:EAX[17:18] を見ていたが、
  これは **CPU が対応する XCR0 ビット**であり、OS が有効にしたかどうかではない。
  XGETBV(0) の同じビットを追加で確認する。5.16 より古い Linux や AMX 非対応の Windows
  では CPUID.D.0 は立つが XCR0 は立たないので、上流は Sapphire Rapids を選んでから
  Linux では arch_prctl で止まり、Windows では AMX 命令で SIGILL になっていた。

### 2.3 エントリポイント側

| 場所 | 変更 |
| --- | --- |
| `interface/gemm.c` | 上流の AMX 許可ブロック(Linux 限定、TU ごとの static フラグ)を `sbgemm_kernels_unavailable()` の呼び出しに置き換える。単精度フォールバックがコンパイルされている TU では呼ぶだけ(置き換えが目的)、無い TU では UNAVAILABLE なら `OPENBLAS_GEMM_STATUS_NO_KERNEL` で戻る。`BGEMM`(bfloat16 出力)は generic カーネルで AMX を使わないので対象外にした(上流は Linux で拒否されると `bgemm` もスキップしていた) |
| `interface/gemm_packed_common.h` | `gemm_packed_request_amxtile()` と TU ごとの static フラグを削除し、`gemm_packed_kernels_ready()` を同関数の薄い包みにする。`gemm_packed_ensure_initialized()` を追加(§3.3) |
| `interface/gemm_pack.c` | `cblas_?gemm_pack_get_size` も `gemm_packed_kernels_ready()` を呼んでからパラメータを読む。単精度フォールバックがある TU では `sbgemm_packed_size` の `sbgemm_float_fallback()` でも解決されるが、無いビルドではこれが唯一の呼び出しになる |
| `driver/level3/gemm_packed.c` | `sbgemm_packed_size` / `_pack` / `_compute` の先頭にある `sbgemm_float_fallback()` が §2.1 の判定を内包するので変更なし |

## 3. ステータス返却

### 3.1 API(`cblas.h`)

```c
int cblas_?gemm_pack(...);      /* MKL では void。文として呼べば互換 */
int cblas_?gemm_compute(...);
int cblas_sbgemm_status(...);   /* cblas_sbgemm と同じ引数、同じ本体 */

#define OPENBLAS_GEMM_STATUS_TOO_LARGE (-1)
#define OPENBLAS_GEMM_STATUS_INTERNAL  (-2)
#define OPENBLAS_GEMM_STATUS_NO_MEMORY (-3)
#define OPENBLAS_GEMM_STATUS_NO_KERNEL (-4)
```

| 戻り値 | 意味 | xerbla |
| --- | --- | --- |
| 0 | 処理した | — |
| 正 | その番号の引数が不正(§3.2 の表の info と同じ。compute で packed バッファが合わなければ 7 / 9) | 呼ぶ |
| `TOO_LARGE` | packed サイズが収まらない(`get_size` が 0 を返す寸法) | info 0 |
| `INTERNAL` | 閉形式サイズと書き込み量の不一致 | info 0 |
| `NO_MEMORY` | 単精度フォールバックの float 展開を確保できなかった | 呼ばない(`openblas_warning(0)`) |
| `NO_KERNEL` | bfloat16 カーネルが動かせず、代替も無い(静的 `SAPPHIRERAPIDS` かつ `BUILD_SINGLE` 無し) | 呼ばない |

失敗時に出力に触れないことは従来どおり。`_pack` は、引数エラーと `TOO_LARGE` では `dest`
に触れず、それ以外の失敗ではヘッダをゼロで潰すので、後続の `_compute` は 7 / 9 で拒否する。
`cblas_sbgemm_status` は `order` が row / column major のどちらでもないとき 1 を返す
(上流の `cblas_sbgemm` は xerbla に 0 を渡す。その値をそのまま返すと成功と区別できない)。

`cblas_sbgemm` の void シグネチャは CBLAS 標準なので変えない。`interface/gemm.c` の
本体は `NAME`(Fortran)と `CNAME` が共有しているため、すべての `return;` を
`GEMM_RETURN(status)` に置き換え、`CBLAS && BFLOAT16 && !BGEMM` の TU だけ
`int GEMM_STATUS_NAME(...)`(= `cblas_sbgemm_status`)を本体にして `void CNAME(...)` を
その包みにする。他の TU では `GEMM_RETURN(x)` は `return` に展開され、引数は評価されない
(`OPENBLAS_GEMM_STATUS_*` は cblas.h の定数で、非 CBLAS TU には見えないが、
使われないマクロ引数は展開されない)。

同じ `interface/gemm.c` の単精度フォールバックは、2 つ目のコミットでは常に Fortran 名の
`sgemm_` を呼んでいた。`ONLY_CBLAS=1`(Makefile では `NO_FBLAS=1`)のビルドには `sgemm_`
が無く、`cblas_sbgemm.o` が未解決シンボルを抱えてリンクに失敗する(本コミットの検証で
見つけた 2 つ目のコミットの欠陥)。CBLAS TU では `cblas_sgemm(CblasColMajor, ...)`、
Fortran TU では `sgemm_` を呼ぶようにした。どちらの TU も、自分と同じインターフェースの
`sgemm` は必ず存在する。

### 3.2 ドライバ(`driver/level3/gemm_packed.c`、`common_level3.h`)

- `?gemm_packed_pack` の戻り値 4(展開の確保失敗)は従来どおり。interface が
  `NO_MEMORY` に写す(以前は `xerbla(info=0)` だった)。
- `?gemm_packed_pack` が 3(閉形式サイズとの不一致)を返すとき、パネルは既に書かれて
  いるのに古いヘッダが残り、以前の pack のヘッダが新しいパネルを説明してしまい得た。
  3 を返す前にヘッダ領域をゼロで潰す。引数エラーと 2(サイズ超過)は何も書く前に
  戻るので `dest` に触れない。
- `?gemm_packed_compute` の戻り値をビットで名付けた: `GEMM_PACKED_COMPUTE_BAD_A` (1)、
  `_BAD_B` (2)、`_NO_MEMORY` (4)。フォールバック側の確保失敗は以前 0(成功)を返して
  いたが、`_NO_MEMORY` を返す。

### 3.3 `get_size` と初期化順序

`cblas_?gemm_pack_get_size` は `blas_memory_alloc` を呼ばない唯一のエントリポイントで、
`DYNAMIC_ARCH` では `gotoblas` のパラメータを直接読む。GCC / clang では
コンストラクタが `gotoblas` を main() 前に埋めるので問題ないが、MSVC にはコンストラクタが
無く、最初の `blas_memory_alloc` で埋まる。`gemm_packed_ensure_initialized()` で
`gotoblas == NULL` なら `gotoblas_dynamic_init()` を呼んでから読む(`_pack` も同じ。
`_compute` は `blas_memory_alloc` を呼ぶので不要)。§2.1 の解決関数も同じ確認をする。

## 4. 変更ファイル

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| 新規 | `driver/others/sbgemm_amx.c` | AMX 可否の解決、`sbgemm_kernels_unavailable()` |
| 変更 | `driver/others/dynamic.c` | `gotoblas_sbgemm_use_cooperlake()`、起動時の置き換え、`force_coretype` の 26、`support_amx_bf16` の XCR0 確認 |
| 変更 | `driver/others/CMakeLists.txt`, `driver/others/Makefile` | `BUILD_BFLOAT16` のとき `sbgemm_amx.c` を組み込む |
| 変更 | `common_sbfallback.h` | 宣言と `sbgemm_float_fallback()` の AMX 判定 |
| 変更 | `interface/gemm.c` | AMX ブロックの置き換え、`GEMM_RETURN`、`cblas_sbgemm_status`、`cblas_sgemm` 経由のフォールバック |
| 変更 | `interface/gemm_pack.c`, `interface/gemm_compute.c`, `interface/gemm_packed_common.h` | `int` 戻り値、ステータス写像、初期化確認、`get_size` の AMX 解決 |
| 変更 | `driver/level3/gemm_packed.c`, `common_level3.h` | compute の `_NO_MEMORY`、ビット名、status 3 のヘッダ無効化 |
| 変更 | `cblas.h` | プロトタイプ、`OPENBLAS_GEMM_STATUS_*`、`cblas_sbgemm_status` |
| 変更 | `exports/gensymbol`, `exports/gensymbol.pl` | `cblas_sbgemm_status` |
| 変更 | `utest/test_extensions/test_{s,d,sb}gemm_pack.c` | ステータスのテスト |
| 変更 | `kernel/setparam-ref.c` | コメントのみ。3 つ目のコミットの「実行時判定は SapphireRapids を報告し得ない」は強制指定の話であり、また本コミットで上限を直したので削った(`TS` を含む語は使っていない) |
| 変更 | `change/` | 本書、差分パッチ、既存 3 書への注記 |

コアごとのコードは追加していない。`sbgemm_amx.c` と `dynamic.c` は 1 回だけコンパイルされる。

## 5. スレッド安全性

- 解決は `blas_lock` で 1 スレッドに限り、他は結果を待つ。SBGEMM フィールドの書き換えは
  ロック内で終え、`MB` の後に state を公開する。x86-64 ではロードの順序が保たれるので、
  READY を見たスレッドは書き換え後の値を読む。
- 書き換わるフィールドを読むのは bfloat16 GEMM のエントリポイントだけで、それらは先に
  この解決を通る。`sgemm` 等が読むフィールドは変わらない。
- 書き換え後に古い SBGEMM フィールドを読み得るのは、この関数を通らない
  `sbgemmt` / `sbgemm_batch` だけ(alinalg は使わない)。上流ではそれらは AMX 許可なしで
  SIGILL になっていたので、退行ではない。

## 6. 対象外

- `cblas_sbgemv`: 上流の `kernel/x86_64/sbgemv_{n,t}.c` は行列の bfloat16 / fp32 コピーを
  `malloc` し、NULL を確認しない。手を入れていない。
- `sbgemmt`、`sbgemm_batch`: AMX 判定を通さない(上流と同じ挙動)。
- `blas_memory_alloc` のスクラッチ: 確保できないときプロセスを終了するのは上流の方針で、
  変更していない。
- Makefile ビルドで AVX512-BF16 組み込み関数を持たないコンパイラ(GCC 10.1 未満)を使うと
  `NO_AVX512BF16=1` になり、Cooperlake / Sapphire Rapids テーブルの `SBGEMM` カーネルが
  generic 2x2 のまま `"Cooperlake"` 等の名前を持つ。名前で判定する
  `sbgemm_float_fallback()` はこのとき単精度フォールバックを選ばない(結果は正しいが遅い)。
  `NO_AVX512BF16` は make 変数で C マクロにならないので、ヘッダから判定できない。

## 7. 検証

環境: AMD Ryzen 7 9800X3D(Zen 5、AVX512-BF16 あり、AMX なし)、Windows 11、
MSYS2 ucrt64 GCC 16.2、CMake 4.4、Ninja、`DYNAMIC_ARCH=ON`。2026-09-22。

| テスト | 内容 | 結果 |
| --- | --- | --- |
| `OPENBLAS_CORETYPE=SapphireRapids` | 起動時に `AMX_FALLBACK` が出て、`openblas_get_corename()` は `"SapphireRapids"` のまま、`cblas_sbgemm_status` が 0 と正しい積(Cooperlake カーネル)を返す。`ldc` 不正で 13 | 期待どおり |
| 利用側(alinalg)gtest | Release 461 件 × コア 6 種(未設定 / Haswell / Zen / SkylakeX / Cooperlake / SapphireRapids)、Debug 488 件(death test 込み)× 4 種(未設定 / Haswell / SapphireRapids / Cooperlake)。`dot` / `dot_compute` の `bool` 化に伴い全呼び出しを `ASSERT_TRUE` に、release の拒否経路 2 件を `EXPECT_FALSE` にした | 全件合格 |
| `openblas_utest_ext` | 695 件 × 同 6 コア。`{s,d,sb}gemm_pack:status_reports_success_and_rejections` を追加(成功 0、引数エラーの番号、`TOO_LARGE`、ヘッダ不一致の 7 / 9、不正な `order` の 1) | 全件合格 |
| 警告 | 変更した 11 オブジェクトを `-Wall -Wextra` で再コンパイル | 新規 0(上流由来の `unused variable 'order'` のみ) |
| Linux | WSL2 Ubuntu、GCC 16.2 で configure し、`sbgemm_amx.c` / `dynamic.c` / `cblas_sbgemm` / `sbgemm` / `cblas_sbgemm_pack` / `cblas_sbgemm_compute` / `sbgemm_packed` をコンパイル(`arch_prctl` 経路が実際にコンパイルされる側) | エラー・警告 0 |
| 静的 `SAPPHIRERAPIDS` ビルドの分岐 | `-UDYNAMIC_ARCH -DSAPPHIRERAPIDS` で `sbgemm_amx.c` / `cblas_sbgemm` / `cblas_sbgemm_pack` / `cblas_sbgemm_compute` / `sbgemm_packed` をコンパイル(リンクなし。`config.h` が Cooperlake 用なので param.h の再定義警告は出るが、新規コードの警告は無い) | コンパイル可 |

Linux の `arch_prctl` 拒否 → 遅延置き換えの経路は、このホストでは実行できない
(AMX の無い CPU では起動時に置き換わるので遅延経路に入らない)。§2.1 のとおり
起動時経路と同じ `gotoblas_sbgemm_use_cooperlake()` を共有するので、未検証なのは
syscall の失敗を UNAVAILABLE 判定へつなぐ数行に限られる。確保失敗の経路も実機では
起こせないため、ステータスの写像は引数エラーとヘッダ不一致で確認した。
