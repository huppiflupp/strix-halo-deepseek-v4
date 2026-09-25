| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |           pp512 |       1142.77 ± 5.79 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |      512 |   1 |          pp2048 |      1101.88 ± 17.65 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |           pp512 |       1092.22 ± 7.51 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |     2048 |   1 |          pp2048 |       1419.81 ± 2.08 |

build: e9f824d8c (529)
