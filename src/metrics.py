from catalyst.callbacks import BatchMetricCallback
from catalyst.utils import metrics
from sklearn.metrics import f1_score
import numpy as np

def sk_learn_f1_score(y_true, sofmax_predict):
    sofmax_predict = sofmax_predict.detach().cpu().numpy()
    f1_score(y_true, np.argmax(sofmax_predict, axis=1), average='macro')

class F1MetricCallback(BatchMetricCallback):
    """F1 score metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f1_score",
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            beta (float): beta param for f_score
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax2d'``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=sk_learn_f1_score,
            input_key=input_key,
            output_key=output_key
        )
        self._reset_stats()

    def _reset_stats(self):
        self.y_true = []
        self.predict = []

    def on_loader_end(self, runner: "IRunner"):
        metric = f1_score(np.array(self.y_true), np.array(self.predict), average='macro')

        runner.loader_metrics[self.prefix] = metric

    def on_batch_end(self, runner: "IRunner") -> None:
        """Computes metrics and add them to batch metrics."""
        output = self._get_output(runner.output, self.output_key)
        input = self._get_input(runner.input, self.input_key)

        sofmax_predict = output.detach().cpu().numpy()
        sofmax_predict = np.argmax(sofmax_predict, axis=1)

        y_true = input.detach().cpu().numpy()


        assert len(input.shape) == 1 and len(sofmax_predict.shape) == 1, (len(input.shape) , len(sofmax_predict.shape))
        assert input.shape == sofmax_predict.shape, (input.shape, sofmax_predict.shape)
        self.y_true.extend(list(y_true))
        self.predict.extend(list(sofmax_predict))
