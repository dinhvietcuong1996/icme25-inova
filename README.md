To run prediction and evaluation on the **validation set**, use the following command:

```python
python inova_pred.py --data-set valid --answers-file answer.jsonl
python inova_compute_metrics.py --answers-file answer.jsonl --result-dir result_dir
```