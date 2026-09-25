| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |           pp512 |       1141.39 ± 7.41 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |          pp2048 |      1101.89 ± 18.14 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |           tg128 |         53.76 ± 0.02 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |           pp512 |       1126.44 ± 8.56 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |          pp2048 |       1421.19 ± 1.71 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |           tg128 |         53.74 ± 0.03 |

build: e9f824d8c (529)
