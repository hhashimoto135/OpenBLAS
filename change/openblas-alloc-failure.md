# alinalg が使う全経路でメモリ確保失敗を報告する

このブランチ(`alinalg-pack-api`)の 5 つ目のコミット。alinalg が呼ぶ OpenBLAS の
機能(`?gemm` / `?gemv` / packed GEMM API / `cblas_sbgemv` / `LAPACKE_?syevd_work` /
`LAPACKE_?syevr_work`)の途中で起こり得る **メモリ確保** を、プロセス終了や
クラッシュや無限リトライではなく、呼び出し側に見える失敗にする。対象は
`DYNAMIC_ARCH` ビルドが選ぶ x86-64 のすべてのコアで、開発機(Zen 5)では実行できない
AMX カーネルも含む。失敗したときの出力の内容は保証しない(カーネルの途中で失敗すると、
出力は途中まで書かれている)。出力を元のまま残すための並べ替えや代替の計算経路は置かない。

上流との差分パッチは同じディレクトリの `openblas-alloc-failure.patch`。
1〜4 つ目のコミットの設計は `openblas-pack-api-strategy.md`、
`openblas-sbgemm-float-fallback.md`、`openblas-zen-blocking-override.md`、
`openblas-amx-fallback-status.md`。

## 1. 背景: どこで確保が起こるか

4 つ目のコミットは単精度フォールバックの float 展開(`malloc`)だけを報告対象にしていた。
alinalg の各機能から到達する確保はそれ以外にもある。

| 確保 | 場所 | 失敗時の上流の挙動 |
| --- | --- | --- |
| 作業バッファ `blas_memory_alloc` | すべての level-2 / level-3 interface(`gemm.c` は `blas_memory_alloc(0)`、`gemv.c` / `ger.c` は `STACK_ALLOC` 経由、`symv.c` / `syr2.c` / `trmv.c` は `(1)`、`syr2k.c` / `trsm.c` は `(0)`) | 領域が `NUM_BUFFERS`(= `NUM_THREADS * 2`、本ビルドで 128)個すべて使用中なら補助配列 512 個を `malloc`(未検査)、それも尽きると「Program is Terminated」を出して **NULL を返す**(呼び出し側は NULL を参照して落ちる)。OS がマッピングを拒否すると 10 回リトライした後に **`exit(1)`**。補助配列側のマッピング失敗は **無限リトライ** |
| 補助配列スロットの再利用 | `memory.c` の `allocation2` | 第 1 配列と違い「領域が既にあるか」を見ないので、スロットを取るたびに **新しい領域をマッピング** し、前の領域(128 MiB)はプロセス終了まで残る。512 回を超えると `new_release_info` の範囲外に書く |
| 小行列カーネル(単精度 / 倍精度 NN) | `kernel/x86_64/{s,d}gemm_small_kernel_nn_skylakex.c`(SkylakeX / Cooperlake / Sapphire Rapids) | 行の端数(倍精度 4 行以下で K ≥ 16、単精度 8 行以下で K ≥ 32)のために A の残りを行優先でコピーする `malloc` が **未検査**。`interface/gemm.c` は作業バッファより前にこの経路へ入るので、NULL なら SIGSEGV。LAPACK の `?laed3` からの `?gemm('N','N')` もここに来る |
| 小行列カーネル(bfloat16) | `kernel/x86_64/sbgemm_small_kernel_template_cooperlake.c`(Cooperlake、AMX の無い Sapphire Rapids) | 呼び出しごとに 64 KiB と 2 MiB を `malloc` し **未検査**。NULL なら C をゼロ/スケールした後に SIGSEGV |
| bfloat16 GEMV のスカラ経路 | `kernel/x86_64/sbgemv_{n,t}.c`(Cooperlake / Sapphire Rapids 以外のコア) | 行列の bfloat16 コピーと fp32 コピー、x の fp32 コピーを毎回 `malloc` し、**NULL を確認しない** |
| bfloat16 GEMV のストライド付きベクトル | 同ファイル、`ALIGN64_ALLOC`(Cooperlake / Sapphire Rapids の高速カーネルは連続ベクトルを要求する) | `malloc` 未検査 |
| 単精度フォールバックの float 展開 | `common_sbfallback.h` | 4 つ目のコミットでステータス返却済み |
| AMX bfloat16 カーネルの一時 C | `kernel/x86_64/sbgemm_kernel_16x16_spr_tmpl.c`(AMX のある Sapphire Rapids、`alpha != 1`) | カーネル呼び出しごとに `malloc` / `calloc` し **未検査**。C は既に beta 倍されている |
| LAPACK 内部の BLAS 呼び出し | `?syevd` / `?syevr` から `?gemm_` / `?gemv_` / `?ger_` / `?symv_` / `?syr2_` / `?syr2k_` / `?trmm_`、および `?larft_lvl2` 経由の `?trmv_`(`lapack-netlib/SRC` のうち本ビルドでコンパイルされるものを静的に辿った閉包。level-1 は確保しない。`?trmv_` へは `k < ilaenv(3, '?LARFT')` のときだけ進み、同梱の f2c 版 `ilaenv` はこれに 0 を返すので実行時には通らないが、Fortran 版 `ilaenv.f` は 64 を返す) | 上の作業バッファと同じ(終了または NULL 参照)。上流のままならそこで止まるが、BLAS 側を「確保できなければ return」にするだけでは BLAS が void なので LAPACK の `info` に伝わらず、**結果が黙って壊れる**。そのためフラグ(§2.3)が要る |

## 2. 設計

### 2.1 作業バッファ: `blas_memory_alloc_try()`(`driver/others/memory.c`)

非 TLS 版の `blas_memory_alloc` 本体を `static blas_memory_alloc_impl(procpos, try_only)` に
し、2 つの薄い包みを置く。

- `blas_memory_alloc(procpos)`: `try_only = 0`。上流どおりの挙動(マッピング失敗で終了、
  領域が尽きたら通知の後に NULL)。変更していないルーチンはこれを使い続ける。
- `blas_memory_alloc_try(procpos)`: `try_only = 1`。確保できないときは **NULL を返す**。
  マッピングに失敗した場合は取得済みのスロット(`memory[position].used`)を
  `blas_memory_free` と同じ手順で返してから戻る。領域が尽きた場合も NULL。
  いずれもスレッドの失敗フラグ(§2.3)を立て、`openblas_warning(0, ...)` を 1 行出す。

補助配列の `malloc` に NULL 確認を足し(両モード)、補助配列側のマッピング失敗の
無限リトライは第 1 配列と同じ 10 回で打ち切る(`try_only` なら NULL、さもなくば上流の
第 1 配列と同じ `exit(1)`)。補助配列のマッピングは第 1 配列と同じく
`if (!newmemory[...].addr)` で囲み、再利用するスロットは自分の領域を使い続ける
(上流のリークと `new_release_info` の範囲外書き込みはこれで無くなる)。

`USE_TLS && SMP` のビルドが使う TLS 版のアロケータの `blas_memory_alloc_try` は
`blas_memory_alloc` を呼び、NULL ならフラグを立てる(TLS 版は表が尽きると通知の後に
NULL を返すので、フラグが無いと try 側の呼び出し元が黙ってスキップすることになる。
その通知は上流の「Program will terminate ...」の 7 行を `printf` で出すが、処理は続く。
マッピング失敗の無限リトライも上流のまま)。alinalg のビルドは非 TLS。

PPC440 のビルドは `memory.c` の代わりに `memory_qalloc.c` を使うので、そこにも
フラグ、`blas_memory_note_failure`、公開関数 2 つ、`blas_memory_alloc_try` を置く
(このアロケータは全呼び出しに同じバッファを 1 つ渡すだけなので、フラグはプロセスに 1 つ)。

### 2.2 呼び出し側: 閉包にある interface だけ切り替える

`blas_memory_alloc` の呼び出し元は interface / driver / lapack に 80 か所近くある。
すべてを NULL 対応にするのではなく、§1 の閉包にある 9 ファイルだけ `blas_memory_alloc_try`
に切り替え、NULL なら return する。

| ファイル | 確保 | 失敗時 |
| --- | --- | --- |
| `interface/gemm.c` | `blas_memory_alloc_try(0)` | `GEMM_RETURN(OPENBLAS_GEMM_STATUS_NO_MEMORY)`(`cblas_sbgemm_status` は値で、他は void で戻る) |
| `interface/gemm_compute.c` | 同 | `OPENBLAS_GEMM_STATUS_NO_MEMORY` |
| `interface/gemv.c`, `interface/ger.c` | 新設の `STACK_ALLOC_TRY`(`common_stackalloc.h`。`MAX_STACK_ALLOC` 以下はスタック、超えると `blas_memory_alloc_try(1)`) | return |
| `interface/symv.c`, `interface/syr2.c`, `interface/trmv.c` | `blas_memory_alloc_try(1)` | return |
| `interface/syr2k.c`, `interface/trsm.c`(`?trmm` も同ファイル) | `blas_memory_alloc_try(0)` | return |

これらのファイルは alinalg が使わないルーチンもビルドするので、そのルーチンも同じように
変わる: `?trsm`、`?her2k`(`syr2k.c`)、`?gemm3m`・`bgemm`・`shgemm`・CMake ビルドの
`?gemmtr`(`gemm.c`)、そして `gemm.c` / `trsm.c` / `syr2k.c` から作られる複素数版
(`c` / `z`)。それ以外のルーチン(`?gemmt`、`?trsv`、`zgemv.c` などの複素数 level-2、
`lapack/` の最適化ドライバなど)は上流のまま。

`gemv.c` / `symv.c` は上流どおり作業バッファを取る前に `y := beta * y` を済ませるので、
確保に失敗すると y は beta 倍された状態で残る。
`cblas_?gemm_compute` はパック済みオペランドのヘッダをバッファを取った後で検査するので、
領域が尽きているときは不正なパックも引数位置(7 / 9)ではなく `NO_MEMORY` で返る。

### 2.3 スレッドごとの失敗フラグ

LAPACK の中で BLAS が return しても `info` には出ない。そこで `memory.c` に
スレッドローカルなフラグを置き、

```c
int  openblas_alloc_failed(void);        /* 0 でなければ、このスレッドの呼び出しが確保失敗で何もせず戻った */
void openblas_clear_alloc_failed(void);
```

を `cblas.h` で公開する(`thread_local` は `driver/level2/gemv_thread.c` と同じ三通り
`_Thread_local` / `__declspec(thread)` / `__thread` で定義する。いずれも無いコンパイラでは
空の定義になり、プロセス共有の 1 フラグになる)。フラグを立てるのは

- `blas_memory_alloc_try` の失敗(§2.1)、
- `blas_memory_note_failure()`(内部関数): `common_sbfallback.h` の float 展開の失敗、
  `sbgemv_{n,t}.c` の `ALIGN64_ALLOC` の失敗、bfloat16 小行列カーネルのブロック確保の失敗(§2.5)。

利用側の作法は「呼ぶ前に clear、呼んだ後に read」。alinalg は `dot` / `dot_prepare_pack` /
`dot_compute` / `eigvalsh` / `eigh` のすべての OpenBLAS 呼び出しでこれを行い、立っていれば
`false` を返す。`cblas_?gemm_pack` / `cblas_?gemm_compute` / `cblas_sbgemm_status` の
戻り値も引き続き `NO_MEMORY` を返す(両方立つ)。単精度フォールバックの中の `sgemm` は
フラグでしか失敗を伝えないので、`interface/gemm.c` はその呼び出しの前後でフラグを読んで
`NO_MEMORY` に写す(呼び出し側が clear していなかった既存のフラグは戻す)。

例外: ARM64 / POWER で `GEMM_GEMV_FORWARD` が有効なビルドでは、m または n が 1 の
`cblas_sbgemm_status` が GEMV に転送され、その GEMV の失敗はフラグにしか出ない
(戻り値は 0)。x86-64 では転送は無効。

`blas_memory_alloc_try` の前に `xerbla` は関与しない: 確保失敗は引数エラーではない。

### 2.4 bfloat16 GEMV(`kernel/x86_64/sbgemv_{n,t}.c`)

- スカラ経路(`HAVE_SBGEMV_{N,T}_ACCL_KERNEL` が無いコア)は、行列と x をコピーせず
  **その場で要素ごとに float へ広げて積和する**よう書き直した。広げる規則は
  `SBF16TOS_K` と同じ(非正規数は同符号のゼロ、NaN は quiet 化。`common_sbfallback.h`
  の `sbgemm_widen` と同じビット演算)。総和の順序も従来と同じ(行ごとに j の昇順)。
  bfloat16 同士の積は float で厳密なので、NaN が現れず各積がゼロか float の正規範囲に
  収まる限り結果はビット一致する。NaN のペイロードと符号は変わり得る(どの NaN が残るかは
  生成コードのオペランド順で決まる)。FMA 付きでコンパイルされるコア(Zen、SkylakeX)では、
  どの積和が融合されるかが変わったので、オーバーフロー / アンダーフローする積を含む行の
  丸めが変わり得る。ストライド付きの x / y もそのまま読み書きするので、この経路の
  確保は **無くなった**。O(m·n) のコピーが無くなるぶん速くもなる。
- 高速カーネル(Cooperlake / Sapphire Rapids)はベクトルの連続性を要求するので、
  ストライド付き x / y の圧縮コピー(`ALIGN64_ALLOC`)は残る。NULL を確認し、
  警告を 1 行出して `blas_memory_note_failure()` を呼んで戻る(`ALIGN64_ALLOC` マクロも
  NULL のとき `ptr_align` を NULL にするよう直した)。

このファイルはコアごとにコンパイルされる共有ソースで、コア固有のコードは追加していない。

### 2.5 小行列カーネル

`interface/gemm.c` は小さな積を作業バッファを取る前に小行列カーネルへ渡す。

- 単精度 / 倍精度 NN(`kernel/x86_64/{s,d}gemm_small_kernel_nn_skylakex.c`): 行の端数の
  内積型の分岐が A の行優先コピー(`mbuf`)を `malloc` する。NULL なら警告を 1 行出し、
  `blas_memory_note_failure()` を呼んで 1 を返す。端数より上の行は既に C に書かれている。
- bfloat16(`kernel/x86_64/sbgemm_small_kernel_template_cooperlake.c`): ブロック A / B の
  `malloc` を確認し、どちらかが NULL なら両方を解放して警告を 1 行出し、
  `blas_memory_note_failure()` を呼んで 1 を返す。
  `interface/gemm.c` の小行列経路はカーネルの戻り値を読み、0 でなければ
  `GEMM_RETURN(OPENBLAS_GEMM_STATUS_NO_MEMORY)` にする。値が意味を持つのは
  `cblas_sbgemm_status` だけで、その小行列カーネルは成功時に 0 を返す。他の関数では
  `GEMM_RETURN` が値を捨てる(LoongArch の LASX DGEMM 小行列カーネルは M を返すが、
  これも捨てられる)。AMX の無い Sapphire Rapids は 4 つ目のコミットでこのカーネルを
  使うので、同じく報告する。`interface/gemm_batch.c` も小行列カーネルを呼ぶが、
  `DYNAMIC_ARCH` ビルドでは表のオフセットをそのまま関数ポインタとして呼ぶので、
  上流の時点で失敗注入をしなくても落ちる。alinalg は使わないのでこのコミットでは触らない。
- AMX(`kernel/x86_64/sbgemm_kernel_16x16_spr_tmpl.c`、AMX のある Sapphire Rapids): `alpha != 1`
  の版はカーネル呼び出しごとに一時 C を `malloc`(`k < 32` なら `calloc`)する。NULL なら
  警告を 1 行出し、`blas_memory_note_failure()` を呼んで 1 を返す。この時点で C は
  ドライバが beta を掛けた後で、それまでのブロックの積も入っている。ドライバはカーネルの
  戻り値を見ないので、`cblas_sbgemm_status` と `cblas_sbgemm_compute` の戻り値には出ず、
  フラグだけが失敗を伝える。

### 2.6 alinalg 側

`include/alinalg/alinalg.h` の OpenBLAS / LAPACKE 呼び出しの前に
`openblas_clear_alloc_failed()`、後に `openblas_alloc_failed()` を見る。

- `dot`(行列×行列、行列×ベクトル、全 3 型): フラグが立っていれば `false`。
  bfloat16 の行列×行列は `cblas_sbgemm_status` の戻り値も見る。
- `dot_prepare_pack`: pack のステータスに加えてフラグ。失敗なら `DotPacked` は無効のまま。
- `dot_compute`(3 オーバーロード): compute のステータスに加えてフラグ。
- `eigvalsh` / `eigh`(3 オーバーロード): `info != 0 || openblas_alloc_failed()` で `false`。

## 3. 上流との違いのまとめ

| 状況 | 上流 | 本フォーク(§2.2 のルーチン) |
| --- | --- | --- |
| 作業バッファのマッピングが 10 回のリトライ後も失敗 | `exit(1)` | 何もせず return、フラグ、警告 |
| 作業バッファが `NUM_BUFFERS + 512` 個すべて使用中 | 「Program is Terminated」を印字して NULL → 呼び出し側で SIGSEGV | 何もせず return、フラグ、警告 |
| 補助配列 512 個の `malloc` 失敗 | NULL 参照 | 上と同じ |
| 補助配列側のマッピング失敗 | 無限リトライ | 10 回で打ち切り(try なら return、さもなくば `exit(1)`) |
| 補助配列スロットの再利用 | 毎回新しい領域をマッピング(リーク、513 回目で範囲外書き込み) | 既存の領域を使う(全ルーチン) |
| 単精度 / 倍精度小行列カーネルの `malloc` 失敗 | NULL 参照 | return、フラグ、警告(C は途中まで) |
| bfloat16 小行列カーネルの `malloc` 失敗 | C を書き換えた後に NULL 参照 | `NO_MEMORY`、フラグ、警告 |
| AMX カーネルの一時 C の確保失敗(`alpha != 1`) | NULL 参照 | return、フラグ、警告(C は途中まで) |
| bfloat16 GEMV スカラ経路の `malloc` 失敗 | NULL 参照 | 確保しない |
| bfloat16 GEMV ストライド付きベクトルの `malloc` 失敗 | NULL 参照 | return、フラグ、警告 |
| LAPACK 内部の BLAS の確保失敗 | 上記のいずれか | BLAS は return、フラグが残り、alinalg が `false` |

警告は失敗した確保ごとに 1 行出る。カーネルがブロックごとに確保する場合は、1 回の呼び出しで
複数行になる。

## 4. 変更ファイル

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| 変更 | `driver/others/memory.c` | `blas_memory_alloc_impl` / `blas_memory_alloc_try`、失敗フラグと公開関数、補助配列の NULL 確認、無限リトライの打ち切り、補助配列スロットの再利用、TLS 版の try |
| 変更 | `driver/others/memory_qalloc.c` | PPC440 用アロケータにフラグ、公開関数、try |
| 変更 | `common.h` | `blas_memory_alloc_try` / `blas_memory_note_failure` / 公開関数の宣言 |
| 変更 | `cblas.h` | `openblas_alloc_failed` / `openblas_clear_alloc_failed`、`NO_MEMORY` の説明 |
| 変更 | `common_stackalloc.h` | `STACK_ALLOC_TRY` |
| 変更 | `interface/gemm.c`, `gemm_compute.c`, `gemv.c`, `ger.c`, `symv.c`, `syr2.c`, `syr2k.c`, `trmv.c`, `trsm.c` | try 版へ切り替え、NULL なら return。`gemm.c` は小行列カーネルの戻り値も見る |
| 変更 | `common_sbfallback.h` | float 展開の失敗でフラグ |
| 変更 | `kernel/x86_64/sbgemv_n.c`, `sbgemv_t.c` | スカラ経路の書き直し、`ALIGN64_ALLOC` の NULL 確認と警告 |
| 変更 | `kernel/x86_64/sgemm_small_kernel_nn_skylakex.c`, `dgemm_small_kernel_nn_skylakex.c` | 端数のコピーの確保の確認、失敗の報告 |
| 変更 | `kernel/x86_64/sbgemm_small_kernel_template_cooperlake.c` | ブロック確保の確認、失敗の報告 |
| 変更 | `kernel/x86_64/sbgemm_kernel_16x16_spr_tmpl.c` | 一時 C の確保の確認、失敗の報告 |
| 変更 | `exports/gensymbol`, `exports/gensymbol.pl` | 公開関数 2 つ |
| 新規 | `utest/test_extensions/test_alloc_failed.c`(+ `utest/CMakeLists.txt`, `utest/Makefile`) | フラグの基本動作 |
| 変更 | `change/` | 本書、差分パッチ |

## 5. スレッド安全性

- `blas_memory_alloc_try` のスロット返却は `blas_memory_free` と同じロック(`alloc_lock`、
  `USE_LOCKING`)の下で行う。
- 失敗フラグはスレッドローカルなので、`USE_THREAD=0 USE_LOCKING=1` で複数スレッドが
  同時に呼んでも互いの失敗は見えない。clear と read を同じスレッドで行うのが前提。
- 領域が尽きたときに待たずに失敗を返すのは設計上の選択である。待てば 640 を超える
  同時呼び出しでも成功し得るが、他スレッドが領域を返す保証は OpenBLAS に無く、
  利用側に `false` を返して判断を委ねる方が alinalg の方針に合う。
- 上流由来の競合が 1 つ残る: 第 1 配列が満杯だと分かったスレッドはロックを一度外してから
  `error:` で取り直すので、その間に別スレッドが補助配列を作ると、空きがあっても
  「空きが無い」として失敗を返し得る(上流ではここで NULL 参照になっていた)。
  128 を超える同時呼び出しでだけ起こる。

## 6. 対象外

- 変更していないルーチン(§2.2 以外)は上流どおり。alinalg からは到達しない。
- スレッド版(`USE_THREAD=1`)のビルドでは、ワーカースレッドで起きた失敗はそのスレッドの
  フラグに立ち、呼び出したスレッドには届かない(`sbgemv` のスレッド版では y の一部だけが
  更新され得る)。スレッド版 level-3 のジョブ配列の `malloc` 失敗は上流どおり `exit(1)`。
  alinalg のビルドは `USE_THREAD=0`。
- `blas_memory_alloc` が最初の呼び出しで行う初期化(`gotoblas_dynamic_init` 等)の中の
  確保は対象外。
- 上流の `blas_shutdown` は `release_pos` を戻さず `new_release_info` も解放しない。
  シャットダウン後にライブラリを使い直す場合だけの問題で、上流のまま。
- LAPACKE 自身の `LAPACKE_malloc` は行優先のときの転置コピーにしか使われず、alinalg は
  列優先で呼ぶので発生しない。発生すれば `LAPACK_WORK_MEMORY_ERROR`(負の `info`)で
  返り、alinalg は既に `false` にしている。
- LAPACK は途中で止まれないので、スキップされた BLAS 呼び出しが残した作業領域の値のまま
  解き終えてから戻る(alinalg はその結果を捨てる)。作業領域に NaN が残っていると
  `?stemr` が終わらないことがあるのは LAPACK 自身の NaN 入力に対する挙動で、意図的に
  NaN で埋めた作業領域でだけ観測した。

## 7. 検証

環境: AMD Ryzen 7 9800X3D(Zen 5)、Windows 11、MSYS2 ucrt64 GCC 16.2、CMake 4.4、Ninja、
`DYNAMIC_ARCH=ON`。2026-09-24。x86-64 の 14 コア(Prescott、Core2、Nehalem、Barcelona、
Sandybridge、Bulldozer、Piledriver、Steamroller、Excavator、Haswell、Zen、SkylakeX、
Cooperlake、SapphireRapids)を `OPENBLAS_CORETYPE` で選び、`OPENBLAS_VERBOSE=2` で実際に
選ばれたコアを確かめた。

この CPU で実行できないものは次のように扱った。

- FMA4 を使う Bulldozer / Piledriver / Steamroller / Excavator のカーネル: 例外ハンドラで
  FMA4 命令をソフトウェアで実行するエミュレータを通した(20 種の命令すべてをビット一致で
  確認済み)。HEAD も同じ箇所で SIGILL になる。
- AMX(SapphireRapids): この CPU では Cooperlake のカーネルに切り替わるので、AMX の
  タイル命令(`_tile_loadconfig` / `loadd` / `stored` / `zero` / `dpbf16ps`)をソフトウェアで
  実行するエミュレータに、HEAD と現行の `sbgemm_kernel_16x16_spr.c` をそれぞれ組み込んだ。

失敗注入ハーネスはリポジトリの外(alinalg の作業用スクラッチ)に置いた。

- 作業バッファの枯渇: Windows のジョブオブジェクトでプロセスのコミット上限を 1.5 GiB にし、
  `blas_memory_alloc_try(0)` を NULL が返るまで呼んで(11 個取った後に拒否)バッファを抱えた
  まま各ルーチンを呼ぶ。バッファを返した後にもう一度呼んで回復を見る。
- `malloc` の失敗: `-Wl,--wrap=malloc` で特定の確保(端数コピーなどの大きさ)だけを失敗させる。
- `memory.c` のフック版: `BUFFER_SIZE` を 1 MiB にし、マッピングと補助配列の `malloc` の
  失敗を注入できるようにした写しをライブラリより先にリンクする。

失敗したときに求めたのは、フラグ(とステータスのある関数では `NO_MEMORY`)が立つこと、
落ちず止まらないこと、確保できるようになった後は全件がビット一致で元の結果に戻ることで、
失敗中の出力の内容は問わない。

| テスト | 内容 | 結果 |
| --- | --- | --- |
| 利用側(alinalg)gtest | Release 470 件、Debug 497 件 × 14 コア + 未設定 | 全件合格 |
| `openblas_utest_ext` | 696 件(`alloc_failed:flag_reflects_calls` を追加)、Release と Debug × 同 | 全件合格 |
| `openblas_utest` | 69 件 × 同 | Haswell、Zen、SkylakeX、Cooperlake、SapphireRapids と未設定で全件合格。Prescott〜Sandybridge と FMA4 の 4 コアでは `?scal` の NaN / Inf の 8〜12 件が落ちるが、HEAD でも同じ件が同じように落ちる |
| netlib の C 版 BLAS テスト | `xscblat1/2/3`、`xdcblat1/2/3`(`ctest/sin*`、`din*`)× 同 | 全ルーチン PASSED。出力は HEAD とバイト一致 |
| 成功時の結果 | 18,700 呼び出し(`?gemm`、`cblas_sbgemm_status`、packed API、`?gemv`、`cblas_sbgemv`、`?ger`、`?symv`、`?syr2`、`?syr2k`、`?trmv`、`?trsm`、LAPACKE)の出力を HEAD のライブラリと比較、14 コア | 全件ビット一致 |
| 作業バッファの枯渇 | C 側 934 ケース、alinalg 側 58 ケース × 同 | 全件でフラグが立ち、落ちない。解放後は全件ビット一致で回復 |
| `malloc` の失敗 | 小行列カーネルの端数コピー(s 約 14,000、d 約 7,000 形状)、bfloat16 小行列(m = 8, 16, 32)、単精度フォールバック、ストライド付き `cblas_sbgemv`、`LAPACKE_?syevd_work`、`alin::dot` / `alin::eigh` × 同 | 全件でフラグ(と `NO_MEMORY`)が立ち、警告が出て、落ちない。リーク 0。解放後は回復 |
| AMX カーネル | HEAD と現行をエミュレータ上で 2,105,984 通りの呼び出し(`im`、`in`、`k` の 16 / 32 での余りをすべて、`k` 1〜1024、`alpha` 1 を含む 4 通り)で比較。一時領域の確保を失敗させた 1,680 通り | 確保できる場合は全件ビット一致。失敗時は 1 を返してフラグが立ち、警告 1 行。`alpha == 1` の版は確保しない |
| フック版 `memory.c` | 補助スロットの再利用、640 個保持からの解放と再取得、補助配列のマッピング失敗と `malloc` 失敗、TLS 版 | 600 回の再利用でマッピング 1 回(上流は毎回、600 回で segfault)、再取得でヒープ破壊なし(上流 0xC0000374)、641 個目で NULL とフラグ。TLS 版の try も NULL でフラグを立てる |
| 警告 | 変更に関わる 104 オブジェクトを現行と HEAD の両方から `-Wall -Wextra` でコンパイル。`common_stackalloc.h` などを経由する残り 3,702 オブジェクトも同様 | 新規 0(`sbgemv_n.c` の未使用変数の警告 12 件が消えた) |
| 構成違い | `memory.c` を USE_LOCKING / ロック無し / SMP / SMP + `USE_TLS=1` / SMP + OpenMP(± TLS)で、`memory_qalloc.c` を Windows の設定と PPC440 形の設定でコンパイル | すべて成功し、try と公開関数を定義している |

Linux(WSL2 Ubuntu 26.04、GCC 15.2、Makefile ビルド、同じ構成)でも現行の作業ツリーと
HEAD をビルドして比べた。

| テスト | 内容 | 結果 |
| --- | --- | --- |
| 警告 | 全 9,092 オブジェクトを `-Wall` と `-Wextra` で現行と HEAD からコンパイル。`MAX_STACK_ALLOC` が定義されるので `STACK_ALLOC_TRY` のスタック側もここでコンパイルされる | 新規 0 |
| テスト | `openblas_utest_ext` 699 件、`openblas_utest` 72 件、netlib の C 版 BLAS テスト × 14 コア + 未設定(FMA4 の 4 コアはエミュレータ上) | 全件合格、HEAD と同じ出力 |
| 作業バッファの枯渇 | overcommit の下で `blas_memory_alloc_try(0)` を 640(128 + 512)個取り、23 ケースを呼ぶ。`ulimit -v 2 GiB` でマッピング失敗も | フラグ(と `NO_MEMORY`)が立ち、スタック側(512 要素 / 2048 バイトまで)は枯渇中も計算する。解放後は全件ビット一致。640 個の取り直しで仮想メモリは増えない(HEAD は 2 巡目で 64 GiB をリークし、3 巡目で SIGSEGV) |
| 成功時の結果 | 112,505 呼び出しを HEAD と比較 × 同 | 全件ビット一致 |

SkylakeX 系のコアでは、`eigh<double>`(n = 300)の結果が HEAD と最後の数ビットだけ違う
ことがあった。原因は上流の `dsyr2k` の結果がバイナリの配置(`.rdata` の位置)で変わること
で、HEAD のライブラリにも無関係な詰め物を入れると同じ差が出る。この変更によるものではない。
