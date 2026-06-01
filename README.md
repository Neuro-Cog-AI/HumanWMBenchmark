# Calculating Surprisal and Memory Update Predictors from CBR-RNN Variants

Example surprisal calculation:

`python3 -m surprisal cclark/cbr-rnn cbrrnn brown.sentitems > brown.surprisal`

Example memory update calculation:

`python3 -m memory_update cclark/cbr-rnn-m cbrrnnm brown.sentitems > brown.memupdate`

CBR-RNN variants on Hugging Face Hub:
- `cclark/cbr-rnn`
- `cclark/cbr-rnn-m`