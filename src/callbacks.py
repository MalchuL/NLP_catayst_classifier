from catalyst.core import Callback, CallbackOrder
import numpy as np
import csv


class PredictionCallback(Callback):
    """F1 score metric callback."""

    def __init__(self, input_key: str = 'Id', output_key: str = "logits", output_csv_path='prediction.csv'):
        super().__init__(CallbackOrder.Logging)
        self.output_csv_path = output_csv_path
        self.output_key = output_key
        self.input_key = input_key
        self._reset_stats()

    def _reset_stats(self):
        self.ids = []
        self.predict = []

    def on_loader_end(self, runner: "IRunner"):

        with open(self.output_csv_path, mode='w') as prediction_csv:
            prediction_writer = csv.writer(prediction_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            prediction_writer.writerow(['Id', 'Predicted'])

            for index, pred in zip(self.ids, self.predict):
                prediction_writer.writerow([index, pred])


    def on_batch_end(self, runner: "IRunner") -> None:
        """Computes metrics and add them to batch metrics."""
        output = runner.output[self.output_key]
        input = runner.input[self.input_key]

        softmax_predict = output.detach().cpu().numpy()
        softmax_predict = np.argmax(softmax_predict, axis=1)

        ids = input.detach().cpu().numpy()


        assert len(input.shape) == 1 and len(softmax_predict.shape) == 1, (len(input.shape) , len(softmax_predict.shape))
        assert input.shape == softmax_predict.shape, (input.shape, softmax_predict.shape)


        self.ids.extend(list(ids))
        self.predict.extend(list(softmax_predict))