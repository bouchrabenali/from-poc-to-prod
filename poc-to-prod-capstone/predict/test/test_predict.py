import unittest
from predict.predict.run import TextPredictionModel


class TestTextPredictionModel(unittest.TestCase):
    def test_predict(self):
        # Setup
        model = TextPredictionModel.from_artefacts("C:/Users/BOUCHRA/Documents/EPF 5A/from poc to prod/poc-to-prod-capstone/poc-to-prod-capstone/train/data/artefacts/2024-01-11-23-08-20")
        # Test
        predictions = model.predict(['example text'])
        # Assert
        self.assertIsNotNone(predictions)


if __name__ == '__main__':
    unittest.main()
