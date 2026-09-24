# Cooperlake の bfloat16 GEMV で、16 列の端数行にも beta を掛ける

このブランチ(`alinalg-pack-api`)の 6 つ目のコミット。Cooperlake / Sapphire Rapids の
`cblas_sbgemv` が、特定の形で一部の行の結果を誤る上流の不具合を直す。
上流との差分パッチは同じディレクトリの `openblas-sbgemv-tail-beta.patch`。
1〜5 つ目のコミットの設計は `openblas-pack-api-strategy.md`、
`openblas-sbgemm-float-fallback.md`、`openblas-zen-blocking-override.md`、
`openblas-amx-fallback-status.md`、`openblas-alloc-failure.md`。

## 1. 症状

内積の長さが 16 で、行列の leading dimension が 16 でなく、出力の長さが 8 の倍数でない
とき(行優先の NoTrans なら 16 列で行間隔が 16 以外、行数が 8 の倍数でない)、最後の 8 の
倍数を超えた行だけ beta が 1 として扱われる。

```c
/* a: 3 x 16 ですべて 1, lda = 20, x = 1, y = {100, 100, 100} */
cblas_sbgemv(CblasRowMajor, CblasNoTrans, 3, 16, 1.0f, a, 20, x, 1, 0.0f, y, 1);
/* 期待値 16 16 16、実際は 116 116 116(beta = 0.5 でも期待値 66 に対して 116) */
```

y の間隔が 1 でなく beta が 0 のときは、`sbgemv_t.c` が y の圧縮コピーを値で埋めずに渡す
ので、その行は初期化されていないヒープの値に足し込まれ、結果が実行のたびに変わる。

Cooperlake と、Cooperlake のカーネルを含む Sapphire Rapids だけで起きる。他のコアは
スカラ経路(5 つ目のコミットで書き直したもの)を通るので正しい。上流の `develop` にも
同じコードがある。alinalg からは、16 列の bfloat16 行列を行間隔 16 以外のビューで渡した
`alin::dot` の行列×ベクトル積がこれに当たり、`true` を返しながら誤った値を出していた。

## 2. 原因

`kernel/x86_64/sbgemv_t_microk_cooperlake_template.c` の `sbgemv_kernel_8x16m_lda*` は、
8 行ずつの本体を `STORE8_COMPLETE_RESULT` で書き、beta と alpha の組み合わせ
(`ZERO_BETA` / `ONE_BETA` / `ONE_ALPHA`)を正しく扱う。残りの行は 1 行ずつ処理するが、
`n == 16` の枝の端数行だけが

```c
y[i] += accum128[0] * alpha;
```

と、どの組み合わせでも y に足し込んでいた。同じ関数の `n < 16` の枝の端数行は、
組み合わせごとに書き分けている。

## 3. 修正

`n == 16` の枝の端数行の書き込みを、`n < 16` の枝と同じ書き分けにした。

```c
#ifndef ZERO_BETA
#ifndef ONE_BETA
                y[i] = alpha * accum128[0] + beta * y[i];
#else
                y[i] = alpha * accum128[0] + y[i];
#endif
#else
#ifndef ONE_ALPHA
                y[i] = accum128[0] * alpha;
#else
                y[i] = accum128[0];
#endif
#endif
```

`ONE_BETA` の場合の結果は従来と同じで、変わるのは beta が 1 でない場合だけ。
`ZERO_BETA` では y を読まなくなるので、初期化されていないコピーの値も使わない。

## 4. 変更ファイル

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| 変更 | `kernel/x86_64/sbgemv_t_microk_cooperlake_template.c` | `n == 16` の端数行の書き込み |
| 新規 | `change/` | 本書、差分パッチ |

## 5. 検証

環境: AMD Ryzen 7 9800X3D(Zen 5)、Windows 11、MSYS2 ucrt64 GCC 16.2、CMake 4.4、Ninja、
`DYNAMIC_ARCH=ON`。2026-09-24。コアの選び方と、FMA4 の 4 コアをエミュレータで動かした
ことは `openblas-alloc-failure.md` の 7 節と同じ。

修正前のライブラリは、HEAD のテンプレートを実際のビルドと同じコマンドでコンパイルした
2 オブジェクト(`sbgemv_t_COOPERLAKE`、`sbgemv_t_SAPPHIRERAPIDS`)を、ビルドした
`libopenblas.a` の写しに差し替えて作った。6,059 メンバーのうち違うのはこの 2 つだけで、
同じ手順で修正後のソースをコンパイルするとビルドのオブジェクトとバイト一致する。

| テスト | 内容 | 結果 |
| --- | --- | --- |
| 修正前後の比較 | m、n が 1〜70 と 100、129、256、257、行優先 / 列優先、NoTrans / Trans、lda の余り 0〜5、incx / incy が 1、2、3、-1、-2、alpha 4 通り、beta 0、1、0.5、-3(beta 0 では y を NaN や ±Inf でも埋める)の 52,569,600 ケース。ライブラリ内の `malloc` を NaN などで埋めて 3 回ずつ、Cooperlake、SapphireRapids、未設定(Cooperlake になる)で | 違うのは 243,750 回の実行で、すべて内積の長さが 16、lda が 16 以外、beta が 1 以外の端数行。8 行単位の本体、y の要素の間の隙間、前後の番兵は変わらず、beta 1 では 1 件も変わらない |
| 正しさ | 同じケースを double で計算した参照と比べる。許容誤差は (K + 3)·2^-24·(\|alpha\|·Σ\|a x\| + \|beta y\|) | 修正後は 0 件(上の 3 設定と Haswell、Prescott、Core2、Nehalem、Barcelona、Sandybridge、Zen、SkylakeX)。修正前は 941,250 出力が誤りで、すべて上記の端数行 |
| 再現性 | 同じ入力を 3 回 | 修正後は全件同じ。修正前は y の間隔が 1 でなく beta 0 の 117,000 件で実行ごとに結果が変わった |
| beta 0 と y の NaN / Inf | 1,146,230,400 出力 | 修正後は NaN / Inf が 1 つも残らない。修正前は端数行に 13 万個あまり残った |
| 利用側の回帰テスト | `OpenBlasDot.Bfloat16MatrixVectorSixteenPaddedColumnsAppliesBeta` を修正前後のライブラリにリンク | 修正前は Cooperlake、SapphireRapids、未設定で 8〜10 行目が失敗(beta を 1 とした 244、260、276)、他のコアでは合格。修正後は全設定で合格 |
| 利用側(alinalg)gtest | Release 471 件、Debug 498 件 × 14 コア + 未設定 | 全件合格 |
| `openblas_utest_ext` | 696 件、Release と Debug × 同 | 全件合格 |
| `openblas_utest` | 69 件 × 同 | Haswell、Zen、SkylakeX、Cooperlake、SapphireRapids と未設定で全件合格。Prescott〜Sandybridge と FMA4 の 4 コアで落ちるのは、HEAD でも落ちる `?scal` の NaN / Inf の 8〜12 件だけ |
| netlib の C 版 BLAS テスト | `xscblat1/2/3`、`xdcblat1/2/3` × 同 | 全ルーチン PASSED |
| 警告 | 2 オブジェクトを `-Wall -Wextra` でコンパイル | `-Wall` では新規 0。`-Wextra` では、`ONE_ALPHA` 版の `sbgemv_kernel_8x16m_lda` が alpha を使わなくなったので未使用引数の警告が 1 件増える。同じ版の他の 19 関数にも同じ警告が出ている |

Linux(WSL2 Ubuntu 26.04、GCC 15.2、Makefile ビルド、同じ構成)でも、5 つ目のコミットまでの
ライブラリと修正後のライブラリをビルドして比べた。

| テスト | 内容 | 結果 |
| --- | --- | --- |
| 正しさ | 84,672 ケースを double の参照と比べる。`MALLOC_PERTURB_` を 8 通りに変えて計 12 回、未設定、Cooperlake、SapphireRapids、SkylakeX、Haswell、Zen で | 修正後は 0 件。修正前は未設定、Cooperlake、SapphireRapids で 718 件 |
| 再現性 | 26,244 ケースの出力を同じ 12 回 × 6 設定で比べる | 修正後は 1 通り。修正前は未設定、Cooperlake、SapphireRapids で 9〜12 通りに分かれた。修正で変わったのは予想どおりの 756 ケースだけで、beta 1 では 0 |
| 端数行 | 内積の長さ 15 / 16 / 17 × 出力の長さ 1〜41 × lda の余り 4 通り × alpha、beta、incx、incy の組み合わせの 177,120 ケース | 修正後は 0 件。修正前は長さ 16 だけで 41,446 件 |
| 1 節の例 | beta 0 / 0.5 / 1 | 修正後は 16 / 66 / 116。修正前は未設定、Cooperlake、SapphireRapids で 3 つとも 116 |
| テスト | `openblas_utest_ext` 699 件、`openblas_utest` 72 件、netlib の C 版 BLAS テスト × 14 コア + 未設定 | 全件合格、HEAD と同じ出力 |
| 警告 | 全オブジェクトを `-Wall` で | 新規 0 |
