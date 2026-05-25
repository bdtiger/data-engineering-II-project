from workerA import add_nums, get_accuracy, get_predictions

from flask import (
   Flask,
   request,
   render_template
)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html', active='home')

@app.route("/accuracy", methods=['POST', 'GET'])
def accuracy():
    score = None
    if request.method == 'POST':
        r = get_accuracy.delay()
        score = r.get()
    return render_template('accuracy.html', active='accuracy', accuracy=score)

@app.route("/predictions", methods=['POST', 'GET'])
def predictions():
    if request.method == 'POST':
        results = get_predictions.delay()
        final_results = results.get()

        r = get_accuracy.delay()
        accuracy = r.get()

        return render_template('result.html', active='predictions',
                               accuracy=accuracy, final_results=final_results)

    return render_template('predictions.html', active='predictions')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)
