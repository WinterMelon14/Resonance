# Resonance

## What is it?

I wanted to try tackling Piano Transcription myself, and made a machine learning model from scratch to accomplish this. It works on online piano recordings as well as my recordings on my own piano. I used CQT with Harmonic Stacking, Attention, and a BiGRU, as well as extensive decoding to accomplish this. Overall, I'm pretty satisfied with the results.

## Comparison with TransKun V2

The final model was evaluated on all 177 recordings in the MAESTRO V3 test split using a Python reimplementation of TransKun’s evaluation protocol. Metrics were computed per recording and then macro-averaged.

| Metric                         |  My P  |  My R  | My F1  | TransKun V2 F1 |    Δ F1 |
| ------------------------------ | -----: | -----: | -----: | -------------: | ------: |
| Activation                     | 0.8352 | 0.8978 | 0.8635 |         0.9530 | −0.0895 |
| Note Onset                     | 0.9049 | 0.8968 | 0.8998 |         0.9832 | −0.0834 |
| Note Onset + Offset            | 0.7123 | 0.7070 | 0.7088 |         0.9349 | −0.2261 |
| Note Onset + Offset + Velocity |    N/A |    N/A |    N/A |         0.9296 |     N/A |
| Pedal Activation               | 0.8322 | 0.9411 | 0.8763 |         0.9541 | −0.0778 |
| Pedal Onset                    | 0.5302 | 0.5516 | 0.5300 |         0.8642 | −0.3342 |
| Pedal Onset + Offset           | 0.4500 | 0.4713 | 0.4517 |         0.8377 | −0.3860 |

The model achieves an onset F1 of **0.8998**, activation F1 of **0.8635**, and pedal-activation F1 of **0.8763**. Its strongest results are therefore note-onset detection and recognizing sustained note and pedal activity. The larger differences in onset+offset and pedal-event metrics show that precise release and pedal-transition timing remains substantially behind TransKun V2.

The velocity-qualified metric is not reported because this model does not predict note velocity; assigning a constant velocity would not constitute a meaningful comparison.

### Evaluation details

* Dataset: MAESTRO V3 test split, 177 recordings
* Aggregation: macro-average over recordings
* Onset tolerance: 50 ms
* Offset tolerance: `max(50 ms, 20% of reference duration)`
* Pitch tolerance: 50 cents
* Reference note durations: sustain-pedal extended and capped at repeated-note onsets
* Activation metrics: continuous interval-duration overlap
* Event metrics: `mir_eval` transcription matching
* Pedal events: evaluated as interval events using the same onset and offset tolerances
* Comparison values: published TransKun V2 MAESTRO V3 results

