# README assets

These files are used by the repository README.

- `salesforce_logo.svg`: the Salesforce logo used on the paper's title page.
  Salesforce and the Salesforce logo are trademarks of Salesforce, Inc.
- `overview.png`: Figure 1 of the paper (benchmark setup and a multi-turn
  funds-transfer demo), rendered from the vector PDF at 2400 px width.
- `leaderboard.png`: the paper's round-1 leaderboard bar chart, rendered from
  the vector PDF at 300 dpi.

Regenerate a PNG from the paper's figure PDF with Poppler, for example:

```sh
pdftoppm -png -singlefile -r 300 completed_model_results_leaderboard.pdf leaderboard
```
