| attack            | policy               |   rmse_mean |   rmse_sd |
|:------------------|:---------------------|------------:|----------:|
| Replay of         | No verification      |      35.748 |     1.811 |
| healthy data      |                      |             |           |
| Degradation       | No verification      |      36.763 |     2.16  |
| masked (drift)    |                      |             |           |
| Random edits      | No verification      |      14.922 |     0.27  |
| (3 sigma)         |                      |             |           |
| Replay of         | Statistical detector |      25.157 |     1.72  |
| healthy data      |                      |             |           |
| Degradation       | Statistical detector |      36.751 |     2.293 |
| masked (drift)    |                      |             |           |
| Random edits      | Statistical detector |      17.708 |     0.547 |
| (3 sigma)         |                      |             |           |
| Replay of         | Ledger verification  |      16.984 |     0.542 |
| healthy data      |                      |             |           |
| Degradation       | Ledger verification  |      16.984 |     0.542 |
| masked (drift)    |                      |             |           |
| Random edits      | Ledger verification  |      16.837 |     0.743 |
| (3 sigma)         |                      |             |           |
| Clean (no attack) | No verification      |      14.635 |     0.204 |