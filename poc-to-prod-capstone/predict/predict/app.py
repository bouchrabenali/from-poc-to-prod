from flask import Flask, request, render_template_string
from predict.predict.run import TextPredictionModel
app = Flask(__name__)
html_form = '''
<!DOCTYPE html>
<html>
<head>
    <title>Text Prediction</title>
</head>
<body>
    <h2>Enter text</h2>
    <form method="post" action="/">
        <textarea name="text" rows="4" cols="50" placeholder="Type your text here..."></textarea>
        <br><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        text_list = [request.form['text']]
        model = TextPredictionModel.from_artefacts('C:/Users/BOUCHRA/Documents/EPF 5A/from poc to prod/poc-to-prod-capstone/poc-to-prod-capstone/train/data/artefacts/2024-01-11-23-08-20')
        predictions = model.predict(text_list)
    return render_template_string(html_form, predictions=predictions)
if __name__ == '__main__':
    app.run(debug=True)


