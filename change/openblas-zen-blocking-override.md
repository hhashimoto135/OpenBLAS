# Zen 4 / Zen 5 ブロッキング上書きを、それを受け取れるコアだけに限定する

このブランチ(`alinalg-pack-api`)の 3 つ目のコミット。`OPENBLAS_CORETYPE` で
AVX512 以外のコアを強制すると `?gemm_kernel` 内で SIGSEGV になる問題を修正する。
上流との差分パッチは同じディレクトリの `openblas-zen-blocking-override.patch`。

1 つ目のコミット(packed GEMM API)の設計は `openblas-pack-api-strategy.md`、
2 つ目(bfloat16 単精度フォールバック)は `openblas-sbgemm-float-fallback.md`。

## 1. 症状

`OPENBLAS_CORETYPE=Haswell` を立てると、素の `cblas_dgemm` が落ちる。

```c
cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            37, 65, 1000, 1.0, a, 37, b, 1000, 0.0, c, 37);
/* -> SIGSEGV in dgemm_kernel_HASWELL */
```

packed API も bfloat16 も関係しない。素の `cblas_sgemm` / `cblas_dgemm` だけで再現する。
`OPENBLAS_CORETYPE` 未設定、`SkylakeX`、`Cooperlake` では起きない。

| 実効コア | sgemm | dgemm |
| --- | --- | --- |
| 未設定(自動 → Cooperlake) / SkylakeX / Cooperlake / Nehalem / SandyBridge | ok | ok |
| Haswell / Zen | ok | **SIGSEGV** |
| Core2 系(Core2, Penryn, Dunnington, Opteron, Opteron_SSE3, Bobcat) | **SIGSEGV** | **SIGSEGV** |
| Prescott 系(Katmai, Coppermine, Northwood, Banias, Athlon, Prescott) | **SIGSEGV** | ok |
| Barcelona | **SIGSEGV** | ok |

## 2. 原因

`kernel/setparam-ref.c` の x86-64 版 `init_parameter()` の末尾に、コアの `#ifdef` で
囲われていないブロックがある。**実行中ホストの** CPUID を見て、AuthenticAMD かつ
`l3_kb % 32768 == 0` かつ `l2_kb == 1024` かつ AVX512F なら、いま初期化している
`TABLE_NAME` の `sgemm_p/q` を 384/512、`dgemm_p/q` を 512/512 に上書きする
(上流 PR #5868 が Zen 4 向けに入れたチューニング)。

これは `force_coretype()`(`driver/others/dynamic.c`)の**後**に走る。つまり
`OPENBLAS_CORETYPE` はカーネルを差し替えるが、ブロッキングはホスト由来のまま残る。

実測(9800X3D = family 0x1A, L2 1024 KB, L3 98304 KB, AVX512F あり):

| `OPENBLAS_CORETYPE` | 実効コア | sgemm p/q | dgemm p/q | dgemm unroll m/n |
| --- | --- | --- | --- | --- |
| 未設定 | Cooperlake | 384/512 | 512/512 | 16/2 |
| Haswell | Haswell | 384/512 | 512/512 | 4/8 |

P/Q は同じでカーネルだけが違う。

溢れるのは `blas_memory_alloc` の `sa` / `sb`(`BUFFER_SIZE` に収まっている)ではなく、
**カーネルが自分のスタックに持つ固定長のパッキングバッファ**である。

```asm
/* kernel/x86_64/dgemm_kernel_4x8_haswell.S */
#ifndef WINDOWS_ABI
#define L_BUFFER_SIZE 256*8*12+4096      /* 28672 B */
#else
#define L_BUFFER_SIZE 128*8*12+512       /* 12800 B */
```

この定数はそのコア自身の `DGEMM_DEFAULT_Q`(`param.h` の HASWELL / ZEN では
`WINDOWS_ABI` で 128、それ以外で 256)に合わせてある。`driver/level3/level3.c` は
カーネルに渡す `k` を `GEMM_Q` で頭打ちにするので、不変条件は
「スタックバッファ ≥ k ≤ GEMM_Q」。`GEMM_Q` が 512 に書き換わるとこれが破れる。

gdb: `dgemm_kernel_HASWELL+370` の `vmovups %ymm2,0x20(%rsi)` で `rsi`=0x5fffe0、
`rsp`=0x5fc000、`r12`=500(k=1000 を Q=512 で割った `min_l`)。自分のフレームの外に書いている。

`k` の閾値を M=N=64 で二分探索すると、最初に落ちる `k` は Haswell dgemm 142、
Zen dgemm 140、Barcelona sgemm 230、Core2 sgemm 270、Prescott sgemm 280。
いずれもそのコア自身の `*GEMM_DEFAULT_Q` の直上で、強制された 512 よりはるかに小さい。

## 3. 実機で起きるか

**起きない。** 上書きの条件は AuthenticAMD **かつ** AVX512F である。

- x86-64-v3 機には定義上 AVX512 がないので条件を満たさない。Intel Haswell はベンダ判定で、
  Zen 1〜3 は AVX512 と `l2_kb == 1024` の両方で外れる。`dgemm_q` はコア本来の
  `DGEMM_DEFAULT_Q` のままになる。
- Windows の `DGEMM_DEFAULT_Q` は 128 で、実測閾値は 142。128 < 142 なのでどの `k` でも安全
  (SysV は 28672/12800 倍で比例し、Q=256 に対して閾値 ≈ 318)。
- Zen 4 / Zen 5 ホストでは上書き自体は走るが、自動選択が Cooperlake に着くので安全。
  ただし `NO_AVX512` ビルドは ZEN に落ちるため、強制なしで落ちる(上流 #6013)。

余裕は薄い。128 に対して閾値 142 は 14 反復しかなく、カーネル側に境界検査はない。

## 4. 修正

上書きを、それを受け取れるコアのテーブルだけに限定する。`setparam-ref.c` は
コアごとに 1 回コンパイルされ、生成される `kernel_config/<CORE>/config_kernel.h` が
`COOPERLAKE` / `SKYLAKEX` / `SAPPHIRERAPIDS` を定義するので、コンパイル時に書ける。

```c
#if defined(COOPERLAKE) || defined(SKYLAKEX) || defined(SAPPHIRERAPIDS)
{
    ... AuthenticAMD + AVX512F なら P/Q を上書き ...
}
#endif
```

修正後(同じホスト):

| `OPENBLAS_CORETYPE` | sgemm p/q | dgemm p/q |
| --- | --- | --- |
| 未設定 / Cooperlake / SkylakeX | 384/512 | 512/512(Zen 4/5 チューニング維持) |
| Haswell / Zen | 320/320 | 512/128(コア本来の値) |

### 4.1 上流 PR #6027 をそのまま取り込まない理由

上流は同じ whitelist を実行時に書いている
(https://github.com/OpenMathLib/OpenBLAS/pull/6027、0.3.35 マイルストーン):

```c
if (strcmp(gotoblas_corename(), "cooperlake") == 0 || ... ) {
```

これは x86-64 では**決して一致しない**。`driver/others/dynamic.c` の `corename[]` は
`"SkylakeX"` / `"Cooperlake"` / `"SapphireRapids"` と大文字を含み、`strcmp` は
大文字小文字を区別する(小文字なのは arm64 の `dynamic_arm64.c` の方)。結果として
クラッシュは直るが、Zen 4 / Zen 5 のチューニングが全コアで失われる。

実行時にコア名で判定すること自体にも難がある。`force_coretype()` のループは
`for (i = 1; i <= 25; i++)`(`dynamic.c:1085`)で、`corename[]` は 27 要素、
`"SapphireRapids"` は添字 26 にある。つまり `OPENBLAS_CORETYPE=SapphireRapids` は
認識されずホスト既定に落ちるので、実行時判定は `"SapphireRapids"` を報告し得ない。

コンパイル時の whitelist はこれらを持たない。

なお `setparam-ref.c` は `DYNAMIC_ARCH` のときしかコンパイルされない
(`kernel/CMakeLists.txt` の `if (${DYNAMIC_ARCH})`、`kernel/Makefile` の
`ifeq ($(DYNAMIC_ARCH), 1)`)。静的ビルドは `driver/others/parameter.c` を使い、
その x86-64 版 `blas_set_parameter()` は `*gemm_p` と `*gemm_r` しか触らず
`*gemm_q` を書かないので、Q は常にコンパイル時の `*GEMM_DEFAULT_Q` のままで、
そもそもこの問題の対象外である。

### 4.2 whitelist に残る既知の露出(本コミットの対象外)

whitelist したコアのカーネルが `q = 512` を受け取れることは、本ビルドが選ぶ C 実装
(`sgemm_kernel_16x4_skylakex_3.c`、`dgemm_kernel_16x2_skylakex.c`)については確認した。
K に比例するスタックバッファを持たない。

ただし clang + Windows では `cmake/utils.cmake` の条件により `.S` 版が選ばれ、
そちらは `L_BUFFER_SIZE` が自分自身の `*GEMM_DEFAULT_Q` に対して既に不足している
(`sgemm_kernel_16x4_skylakex.S` は Windows で 8192 B、自分の `SGEMM_DEFAULT_Q` 448 には
2,560 B 足りない。`dgemm_kernel_16x2_skylakex.S` は半分あたり 8192 B、自分の
`DGEMM_DEFAULT_Q` 384 には 1,024 B 足りない)。これは Zen ブロッキングとは独立した
上流の採寸ミスで、本ビルド(GCC)では到達しない。上流に報告すべき別件。

関連する上流の issue: #6013(0.3.34 の退行)、#6021(強制コアに適用される)、
#6026(`dgemm_kernel_HASWELL` の大きい k でのスタック溢れ)。いずれも
v0.3.34 には入っていない(2026-09 時点の最新リリースは v0.3.34)。

## 5. 検証

- 素の `cblas_sgemm` / `cblas_dgemm`: §1 の表の全コア × 全形状で SIGSEGV 解消。
- 利用側(alinalg)gtest 460 件を、未設定 / Haswell / Zen / SkylakeX / Cooperlake /
  Nehalem / SandyBridge / Core2 / Prescott / Barcelona の 10 コアで全件合格。
  修正前は Haswell / Zen で `OpenBlasDotPack.LargeKCrossesBlocks` などが落ちていた。
- P/Q の実測値が §4 の表どおりであること(Cooperlake / SkylakeX でチューニングが残ること)を
  `gotoblas` のダンプで確認。
- bfloat16 フォールバックの数値一致と性能に変化がないことを確認
  (768³ で 0.91x / 0.96x、bf16 全 65536 パターンでビット一致)。

- 呼び出し側フレームに 2 KiB のカナリアを置いた 10 コア × 6 形状の走査で、
  最大誤差 0、カナリア破壊 0。`sa` + `sb` が `BUFFER_SIZE` に収まることも全コアで確認
  (最悪 SkylakeX dgemm の 134,152,192 ≤ 134,217,728)。ヒープ側は元から溢れていない。
- `gemm` の nn / tn / nt に加えて `syrk` / `syr2k` / `trmm` / `trsm` も 11 コア × 3 形状で確認。

Bulldozer 系(Bulldozer / Piledriver / Steamroller / Excavator)を強制すると
このホストでは FMA4 / XOP が無いため SIGILL になる(`vfmaddpd`)。これは想定どおりで、
本件とは別。修正前はパッキングループで先に落ちていたので SIGSEGV に見えていた。

## 6. このファイルを編集するときの注意

`kernel/CMakeLists.txt` は `setparam-ref.c` を読み込んで **ファイル全体に対して**
`string(REPLACE "TS" "${TSUFFIX}")` をかけてからコアごとにコピーする
(Makefile ビルドも `sed` で同じことをする)。部分文字列 `TS` は文字列中でもコメント中でも
置換される。既存の被害が生成物の 27 行目に見える(`PROFITS` → `PROFI_HASWELL`)。
本コミットで追加したコメントに `TS` は含まれず、生成物でも原文のまま残ることを確認した。
以後の編集でも `TS` を含む語を書かないこと。
