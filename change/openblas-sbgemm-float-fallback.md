# bfloat16 カーネルを持たない x86-64 コアでの SBGEMM 単精度フォールバック

このブランチ(`alinalg-pack-api`)の 2 つ目のコミット。専用 bfloat16 GEMM カーネルを
持たない x86-64 コアで、`SBGEMM` と packed SBGEMM API をオペランドの float 展開 +
`SGEMM` カーネルに置き換える。1 つ目のコミット(packed GEMM API)の設計は同じ
ディレクトリの `openblas-pack-api-strategy.md`、上流との差分パッチは
`openblas-sbgemm-float-fallback.patch`(このディレクトリ自身は含まない)。

## 1. 背景

`kernel/Makefile.L3` は、コアの `KERNEL.*` が `SBGEMMKERNEL` を定義しない限り

```make
SBGEMMKERNEL    = ../generic/gemmkernel_2x2.c
SBGEMMINCOPY    = ../generic/gemm_ncopy_2.c
SBGEMMITCOPY    = ../generic/gemm_tcopy_2.c
SBGEMMONCOPY    = ../generic/gemm_ncopy_2.c
SBGEMMOTCOPY    = ../generic/gemm_tcopy_2.c
```

を使う。x86-64 でこれを定義しているのは `KERNEL.COOPERLAKE`(AVX512-BF16 の
`vdpbf16ps`)と `KERNEL.SAPPHIRERAPIDS`(AMX-BF16)だけである。したがって
Haswell / Zen / SkylakeX などが選ばれると、`SGEMM` が調整済みアセンブリカーネルを
回す一方で `SBGEMM` は完全スカラーの 2x2 C カーネルを回すことになり、性能差は
一桁になる。

`kernel/generic/gemmkernel_2x2.c` は `conversion_macros.h` の `TO_F32()` で要素ごとに
float へ広げてから乗算しているので、この経路が bfloat16 のまま計算しているわけでは
ない。`TO_F32()` の実体は `sbf16tos_` である。したがって両オペランドを同じ規則で
あらかじめ float へ展開して `SGEMM` カーネルに渡せば、計算する積は同じものになる。

「同じ規則で」が要点である。正規数の拡張は符号・指数がそのままで仮数がゼロ拡張される
可逆変換だが、`SBF16TOS_K`(`kernel/x86_64/bf16to.c`)はそれに加えて **非正規数を同符号の
ゼロに潰し、signaling NaN を quiet NaN にする**。フォールバック側の展開もこの 2 つを
再現する(§5.1)。

## 2. 判定条件

**「CPU が AVX512-BF16 を持つか」ではなく「実行時に選択されたコアが bfloat16 カーネルを
持つか」** で判定する。`OPENBLAS_CORETYPE` でコアを固定すると、bfloat16 対応 CPU 上でも
カーネルは generic のままになり得るためである。

`common_sbfallback.h` の `sbgemm_float_fallback()`:

- `DYNAMIC_ARCH`: `gotoblas_corename()` が `"Cooperlake"` / `"SapphireRapids"` 以外なら
  フォールバックする。
- 非 `DYNAMIC_ARCH`: `COOPERLAKE` / `SAPPHIRERAPIDS` マクロの有無で決める。

`gotoblas` はプロセス初期化時に一度だけ決まるので、判定はプロセス内で不変である。
pack と compute で結果が食い違うことはない。

有効になる条件は `BFLOAT16 && !BGEMM && ARCH_X86_64 && BUILD_SINGLE`。`BGEMM`
(bfloat16 入出力)は対象外、`BUILD_SINGLE` が無いと `sgemm` 側のオブジェクトが
存在しない。

## 3. 変更ファイル

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| 新規 | `common_sbfallback.h` | 判定 `sbgemm_float_fallback()`、float 展開 `sbgemm_expand_to_float()`、先頭次元 `sbgemm_expanded_ld()` |
| 変更 | `interface/gemm.c` | 素の `SBGEMM` を float 展開 + `sgemm_` に置き換え |
| 変更 | `driver/level3/gemm_packed.c` | `sbgemm_packed_{size,pack,compute}` を `sgemm_packed_*` へ委譲 |
| 変更 | `interface/gemm_compute.c` | フォールバック時のスクラッチ分割を `SGEMM_P * SGEMM_Q * sizeof(float)` に |
| 変更 | `interface/gemm_pack.c` | 新しいステータス 4 のコメント |
| 新規 | `change/` | 本書と差分パッチ(ビルドには関与しない) |

コアごとのコードは追加していない。`DYNAMIC_ARCH` でどのコアが選ばれても同じ経路を通る。

## 4. 素の SBGEMM(`interface/gemm.c`)

`NAME`(Fortran)と `CNAME`(CBLAS)は引数検査の後に合流し、以降は共通の本体を共有する。
挿入点はその共通部、引数検査と AMX 権限取得の後である。両エントリポイントが 1 か所で
拾われる。

```c
if (sbgemm_float_fallback()) {
    /* args は内部 column-major。CblasRowMajor なら CNAME が A/B と m/n を入れ替え済み */
    a_rows = transa ? args.k : args.m;   a_cols = transa ? args.m : args.k;
    b_rows = transb ? args.n : args.k;   b_cols = transb ? args.k : args.n;
    ... sbgemm_expand_to_float() で詰めた column-major の float 行列を作る ...
    BLASFUNC(sgemm)(&ta, &tb, &m, &n, &k, &alpha, a_float, &a_rows, b_float, &b_rows,
                    &beta, c, &ldc);
}
```

`alpha` / `beta` / `C` は `SBGEMM` も `SGEMM` も float なので、そのまま渡せる。

`k == 0` または `alpha == 0` のときは展開しない。`driver/level3/level3.c` はこの 2 つの
場合、beta を適用した後 A も B も読まずに戻るので、展開は丸ごと無駄になる(20000 角の
行列なら 3.2 GB の確保と変換が `C := beta * C` のためだけに走ることになる)。
オペランドを空のまま `k = 0` で `sgemm_` を呼ぶと、ドライバが読まないことが保証される
経路に入る。

なお x86-64 では `GEMM_GEMV_FORWARD` が定義されない(`cmake/system.cmake` は arm64 /
riscv / power でのみ立てる)ので、`interface/gemm.c` の GEMM → GEMV 転送はそもそも
プリプロセスで消える。`m == 1` / `n == 1` の `SBGEMM` も level3 ドライバへ行くため、
フォールバックが拾う。

## 5. packed API

`sbgemm_packed_*` を `sgemm_packed_*` へ委譲する。packed バッファは float パネルと
`SGEMM` のブロッキングパラメータを持つことになる。

| 関数 | フォールバック時の動作 |
| --- | --- |
| `sbgemm_packed_size` | `sgemm_packed_size` をそのまま返す(float パネルの大きさが要る) |
| `sbgemm_packed_pack` | src を float へ展開して `sgemm_packed_pack`、その後ヘッダの type_tag を `"sb"` に打ち直す |
| `sbgemm_packed_compute` | packed でない側を float へ展開して `sgemm_packed_compute`、期待 type_tag は `"sb"` |

`sbgemm_packed_compute` のフォールバックは **sb 側のヘッダ検証より前** に return する。
バッファを書いたのは `sgemm_packed_pack` なので、`data_offset` や `data_size` を
検証できるのは `sgemm_packed_compute` だけだからである。

ただしそれだけだと type_tag が `"s"` のままになり、`cblas_sgemm_pack` が作った
float のバッファを `cblas_sbgemm_compute` が受理してしまう(逆も同様)。Cooperlake では
拒否され、Haswell では通る、というコア依存の差になる。そこで
**期待する type_tag を `?gemm_packed_compute` の引数にした**。`sbgemm_packed_pack` は
`sgemm_packed_pack` が書いたヘッダの type_tag を `"sb"` に上書きし、
`sbgemm_packed_compute` は `"sb"` を要求して `sgemm_packed_compute` に渡す。
`gemm_packed_header_check` はそれと比較する。これで

- `cblas_sbgemm_compute` は `cblas_sgemm_pack` のバッファを拒否する
- `cblas_sgemm_compute` は `cblas_sbgemm_pack` のバッファを拒否する

の両方が、フォールバックの有無にかかわらず成り立つ。type_tag の定数は
`common_level3.h` の `GEMM_PACKED_TAG_S` / `_D` / `_SB` に移した。

`interface/gemm_compute.c` のスクラッチ分割は変更が必要である。`sa` / `sb` は
`blas_memory_alloc(0)` の `BUFFER_SIZE` バッファを

```c
sb = sa + GEMM_P * GEMM_Q * COMPSIZE * SIZE
```

で 2 分割したものだが、`SBGEMM` の翻訳単位では `SIZE` が 2(bfloat16)である。
float パネルを書く `SGEMM` カーネルには足りないので、フォールバック時は
`SGEMM_P * SGEMM_Q * sizeof(float)` で分ける。これは `SGEMM` としてコンパイルされた
`interface/gemm.c` が使う分割と同一である。

## 6. メモリと失敗時の挙動

float 展開は `malloc` で確保する。素の `SBGEMM` では A と B の両方、packed では
packed でない側だけなので、一時領域は最大 `(m*k + k*n) * 4` バイト、bfloat16 の
入力の 2 倍である。上流の `kernel/x86_64/sbgemv_{n,t}.c` のフォールバックが行列 1 枚
あたり bfloat16 と fp32 の 2 本を確保するのと同じ性質の割り切りで、それより少ない。

確保に失敗したときは、**`C` に触れずに `openblas_warning(0, ...)` を出して戻る**。

- `cblas_sbgemm`: 何も書かずに戻る。
- `cblas_sbgemm_pack`: ヘッダ領域をゼロで潰して(後続の compute が確実に拒否する)
  ステータス 4 を返す。
- `cblas_sbgemm_compute`: `C` は変更しない。

> 4 つ目のコミット(`openblas-amx-fallback-status.md`)で、この失敗は呼び出し側にも
> 伝わるようになった。`cblas_sbgemm_pack` / `cblas_sbgemm_compute` は
> `OPENBLAS_GEMM_STATUS_NO_MEMORY` を返し(xerbla は呼ばない)、`cblas_sbgemm` は
> void のままで、同じ本体を持つ `cblas_sbgemm_status` が同じ値を返す。
> 判定条件(§2)にも AMX の可否が加わった: Sapphire Rapids で AMX が使えないとき、
> `DYNAMIC_ARCH` では Cooperlake テーブルへ切り替わるので本フォールバックは
> 関与せず、静的 `SAPPHIRERAPIDS` ビルドでは `sbgemm_float_fallback()` が 1 を返して
> 本フォールバックが引き受ける。

## 7. 数値差

展開規則が `SBF16TOS_K` と同一(§1)なので、カーネルが見る値は従来の generic 2x2 経路と
同じであり、精度が落ちることはない。ただし総和順序は
カーネルのブロッキングとアンローリングで決まるため、

- generic 2x2 カーネル
- Cooperlake の `vdpbf16ps`(2 要素ずつ積和)
- 本フォールバック(`SGEMM` カーネル)

の 3 つは互いにビット一致しない。丸め誤差の範囲で一致する。BLAS がコアごとに結果を
変えるのは通常の性質であり、同梱 utest の `compare_sgemm_sbgemm.c` も許容誤差付きで
比較している。

なお `dot` と `dot_compute`(素の SBGEMM と packed SBGEMM)は、フォールバックが有効な
ときは両方が `SGEMM` カーネルを通るので、従来どおり互いに丸めの範囲で一致する。

## 8. 対象外

- **`SBGEMV`**: `cblas_sbgemv` を直接呼ぶ経路は手を入れていない。非 Cooperlake では
  依然として `kernel/x86_64/sbgemv_{n,t}.c` のスカラー経路で、呼び出しごとに行列の
  bfloat16 コピーと fp32 コピーを malloc する。x86-64 では `SBGEMM` からここへ転送される
  ことはない(§4)ので、影響を受けるのは `cblas_sbgemv` の直接呼び出しだけである。
- **`SBGEMMT`**: 同様に対象外。
- **`BGEMM`**(bfloat16 入出力): 出力も bfloat16 なので `SGEMM` にそのままは委譲できない。
- **x86-64 以外**: POWER10 / Neoverse N2 / RISC-V ZVL は bfloat16 カーネルを持つ。
  それ以外のアーキテクチャの generic フォールバックは本フォークの対象範囲外。

## 9. 検証

判定がコア依存なので、`OPENBLAS_CORETYPE` で両方の経路を踏む。

```sh
OPENBLAS_CORETYPE=Cooperlake ...   # 従来の bfloat16 カーネル
OPENBLAS_CORETYPE=Haswell    ...   # フォールバック
```

AVX512-BF16 を持つホスト(Zen4 / Zen5 / Cooperlake / Sapphire Rapids)では
`DYNAMIC_ARCH` が既定で `Cooperlake` を選ぶため、何も指定しないとフォールバック経路は
再現しない。

なお `OPENBLAS_CORETYPE` で AVX512 以外のコアを強制すると、このホストでは
本コミット以前は素の `?gemm` が SIGSEGV になっていた。3 つ目のコミットで修正済み
(`openblas-zen-blocking-override.md`)。
