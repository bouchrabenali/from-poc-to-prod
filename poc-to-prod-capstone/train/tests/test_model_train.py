from unittest.mock import MagicMock
import tempfile
import unittest
import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    #utils.LocalTextCategorizationDataset.load_dataset = MagicMock(side_effect=load_dataset_mock)
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())
    def test_train(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            'batch_size': 2,
            'epochs': 5,
            'dense_dim': 64,
            'min_samples_per_label': 4,
            'verbose': 1
        }

     # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
         # run a training
            accuracy, _ = run.train('chemin_fictif_pour_dataset', params, model_dir, False)


    # TODO: CODE HERE
    # assert that accuracy is equal to 1.0
        self.assertEqual(accuracy, 1.0)



