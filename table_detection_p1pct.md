| attack        | method                      |   recall |   recall_sd |   precision |     fpr |
|:--------------|:----------------------------|---------:|------------:|------------:|--------:|
| modify_1sigma | Merkle-on-chain             |    1     |       0     |       1     |   0     |
| modify_1sigma | Statistical + rules         |    0.004 |       0.004 |       0.014 |   0.003 |
| modify_1sigma | Central checksum (outsider) |    1     |       0     |     nan     | nan     |
| modify_1sigma | Central checksum (insider)  |    0     |       0     |     nan     | nan     |
| delete        | Merkle-on-chain             |    1     |       0     |       1     |   0     |
| delete        | Statistical + rules         |    0.991 |       0.008 |       0.786 |   0.003 |
| delete        | Central checksum (outsider) |    1     |       0     |     nan     | nan     |
| delete        | Central checksum (insider)  |    0     |       0     |     nan     | nan     |
| insert        | Merkle-on-chain             |    1     |       0     |       1     |   0     |
| insert        | Statistical + rules         |    1     |       0     |       0.44  |   0.013 |
| insert        | Central checksum (outsider) |    1     |       0     |     nan     | nan     |
| insert        | Central checksum (insider)  |    0     |       0     |     nan     | nan     |
| replay        | Merkle-on-chain             |    1     |       0     |       1     |   0     |
| replay        | Statistical + rules         |    0.024 |       0.014 |       0.073 |   0.003 |
| replay        | Central checksum (outsider) |    1     |       0     |     nan     | nan     |
| replay        | Central checksum (insider)  |    0     |       0     |     nan     | nan     |