# FluxAnchor metric diagnostics

Data file: `/home/huangjiesheng/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv`

Classic genes: Zwf, Pgi, Crp

## Aggregate classic-mutant baseline

| Method | raw_r2 | log1p_r2 | log10eps_r2 | raw_rmse | log1p_rmse | zero_true_fraction | lt_0p1_fraction | lt_0p5_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FluxAnchor | 0.5879 | 0.7711 | 0.5816 | 1.0655 | 0.2561 | 0.3598 | 0.5344 | 0.7460 |
| KinLLM | 0.5773 | 0.7377 | 0.5299 | 1.0791 | 0.2742 | 0.3598 | 0.5344 | 0.7460 |
| fba | 0.4322 | 0.6181 | 0.6006 | 1.2507 | 0.3308 | 0.3598 | 0.5344 | 0.7460 |


## Calibration trained on non-classic mutants, tested on classic mutants

| scope | Method | raw_r2 | log1p_r2 | raw_rmse | log1p_rmse | calibration_slope | calibration_intercept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_classic_test | FluxAnchor | 0.6299 | 0.7424 | 1.0098 | 0.2717 | 0.7124 | 0.1907 |
| classic_Zwf_test | FluxAnchor | 0.5936 | 0.8221 | 1.4903 | 0.2591 | 0.7124 | 0.1907 |
| classic_Pgi_test | FluxAnchor | 0.7955 | 0.7967 | 0.5135 | 0.2111 | 0.7124 | 0.1907 |
| classic_Crp_test | FluxAnchor | 0.6011 | 0.5795 | 0.7580 | 0.3313 | 0.7124 | 0.1907 |
| all_classic_test | KinLLM | 0.6176 | 0.6983 | 1.0264 | 0.2940 | 0.6764 | 0.2394 |
| classic_Zwf_test | KinLLM | 0.6046 | 0.8126 | 1.4701 | 0.2659 | 0.6764 | 0.2394 |
| classic_Pgi_test | KinLLM | 0.7827 | 0.7573 | 0.5294 | 0.2307 | 0.6764 | 0.2394 |
| classic_Crp_test | KinLLM | 0.5008 | 0.4812 | 0.8479 | 0.3680 | 0.6764 | 0.2394 |
| all_classic_test | fba | 0.5430 | 0.6163 | 1.1221 | 0.3316 | 0.6584 | 0.2343 |
| classic_Zwf_test | fba | 0.5062 | 0.6513 | 1.6429 | 0.3626 | 0.6584 | 0.2343 |
| classic_Pgi_test | fba | 0.7015 | 0.6601 | 0.6204 | 0.2730 | 0.6584 | 0.2343 |
| classic_Crp_test | fba | 0.5186 | 0.5259 | 0.8326 | 0.3518 | 0.6584 | 0.2343 |


## Baseline threshold-free metric per classic gene

| Gene | Method | raw_r2 | log1p_r2 | log10eps_r2 |
| --- | --- | --- | --- | --- |
| Zwf | FluxAnchor | 0.5649 | 0.8635 | 0.6710 |
| Zwf | KinLLM | 0.5710 | 0.8714 | 0.6785 |
| Zwf | fba | 0.4417 | 0.6565 | 0.6082 |
| Pgi | FluxAnchor | 0.7902 | 0.8733 | 0.6349 |
| Pgi | KinLLM | 0.7899 | 0.8717 | 0.6028 |
| Pgi | fba | 0.6435 | 0.7281 | 0.6662 |
| Crp | FluxAnchor | 0.4741 | 0.5499 | 0.4389 |
| Crp | KinLLM | 0.3907 | 0.4298 | 0.3085 |
| Crp | fba | 0.1795 | 0.4672 | 0.5283 |


## Top FluxAnchor error-contributing reactions

| Gene | Reaction | true | pred | abs_error | sq_error | log1p_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| Crp | FUM | 3.4676 | 0.3859 | 3.0818 | 9.4975 | 1.1705 |
| Crp | PFK | 2.9938 | 0.0000 | 2.9938 | 8.9630 | 1.3847 |
| Crp | MDH | 3.2864 | 0.3844 | 2.9020 | 8.4217 | 1.1302 |
| Crp | SUCOAS_reverse | 2.6417 | 0.0000 | 2.6417 | 6.9785 | 1.2924 |
| Crp | FBA | 2.6380 | 0.0000 | 2.6380 | 6.9593 | 1.2914 |
| Crp | PGM_reverse | 4.2659 | 6.6563 | 2.3904 | 5.7139 | 0.3743 |
| Crp | ENO | 4.2690 | 6.6563 | 2.3873 | 5.6990 | 0.3737 |
| Crp | PGL | 3.1502 | 0.7823 | 2.3679 | 5.6068 | 0.8452 |
| Crp | G6PDH2r | 3.1502 | 0.7823 | 2.3679 | 5.6068 | 0.8452 |
| Crp | TPI | 0.8183 | 3.1525 | 2.3342 | 5.4484 | 0.8258 |
| Crp | GAPD | 4.9991 | 7.2929 | 2.2938 | 5.2613 | 0.3238 |
| Crp | PGK_reverse | 4.9991 | 7.2929 | 2.2938 | 5.2613 | 0.3238 |
| Pgi | PTAr | 4.5656 | 1.3101 | 3.2555 | 10.5982 | 0.8793 |
| Pgi | ACKr_reverse | 4.5656 | 1.3101 | 3.2555 | 10.5982 | 0.8793 |
| Pgi | GAPD | 5.1922 | 6.6243 | 1.4321 | 2.0510 | 0.2081 |
| Pgi | PGK_reverse | 5.1922 | 6.6243 | 1.4321 | 2.0510 | 0.2081 |
| Pgi | PGM_reverse | 4.8109 | 6.0349 | 1.2241 | 1.4983 | 0.1912 |
| Pgi | ENO | 4.8131 | 6.0349 | 1.2218 | 1.4927 | 0.1908 |
| Pgi | TPI | 1.9247 | 3.0929 | 1.1683 | 1.3648 | 0.3361 |
| Pgi | MDH | 1.9300 | 1.1160 | 0.8140 | 0.6626 | 0.3255 |
| Pgi | PPCK | 0.6762 | 0.0000 | 0.6762 | 0.4573 | 0.5165 |
| Pgi | GND | 2.2870 | 1.7099 | 0.5772 | 0.3331 | 0.1931 |
| Pgi | EDA | 0.5507 | 0.0000 | 0.5507 | 0.3033 | 0.4387 |
| Pgi | EDD | 0.5507 | 0.0000 | 0.5507 | 0.3033 | 0.4387 |
| Zwf | ACKr_reverse | 14.1647 | 4.3351 | 9.8297 | 96.6221 | 1.0447 |
| Zwf | PTAr | 14.1647 | 4.3351 | 9.8297 | 96.6221 | 1.0447 |
| Zwf | PGM_reverse | 6.5046 | 10.6825 | 4.1778 | 17.4543 | 0.4426 |
| Zwf | GAPD | 7.3451 | 11.5228 | 4.1776 | 17.4528 | 0.4059 |
| Zwf | PGK_reverse | 7.3451 | 11.5228 | 4.1776 | 17.4528 | 0.4059 |
| Zwf | ENO | 6.5174 | 10.6825 | 4.1650 | 17.3476 | 0.4409 |
| Zwf | TPI | 1.8327 | 5.8551 | 4.0224 | 16.1801 | 0.8838 |
| Zwf | GLCptspp | 4.5204 | 6.3500 | 1.8296 | 3.3476 | 0.2863 |
| Zwf | PFK | 5.2246 | 3.4446 | 1.7800 | 3.1686 | 0.3368 |
| Zwf | FBA | 5.2084 | 3.4446 | 1.7638 | 3.1110 | 0.3342 |
| Zwf | PGI | 4.6956 | 6.3500 | 1.6544 | 2.7370 | 0.2550 |
| Zwf | MDH | 2.5210 | 1.2025 | 1.3185 | 1.7384 | 0.4691 |


## Notes

- `log1p` is the most stable log-style metric here because many reactions are exactly zero; `log10(x + 1e-6)` can become misleadingly harsh or unstable under thresholding.

- Any calibration result should be interpreted only when trained on non-classic mutants and evaluated on classic mutants. In-sample calibration is not reported as evidence of real improvement.

- Filtering by higher true-flux thresholds did not reliably improve raw `R^2` for FluxAnchor on these classic mutants.
