| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |           pp512 |       1145.48 ± 5.54 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |          pp2048 |      1103.84 ± 16.97 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |           pp512 |       1095.07 ± 7.04 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |          pp2048 |       1425.49 ± 2.77 |

build: e9f824d8c (529)
