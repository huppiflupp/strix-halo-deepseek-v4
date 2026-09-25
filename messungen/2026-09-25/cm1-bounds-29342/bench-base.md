| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |           pp512 |       1157.58 ± 6.27 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |          pp2048 |      1113.62 ± 17.98 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |           tg128 |         53.71 ± 0.01 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |           pp512 |      1133.24 ± 10.25 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |          pp2048 |       1430.74 ± 5.58 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |           tg128 |         53.70 ± 0.01 |

build: e9f824d8c (529)
