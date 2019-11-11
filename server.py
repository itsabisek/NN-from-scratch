from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('NeuralNet_Main.html')


if __name__ == "__main__":
    app.run(port=5555)