from flask import Flask, render_template, request


# App config.
app = Flask(__name__)

@app.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return render_template('index.html')


@app.route('/hello/<name>/')
def hello(name):
    """ Displays the page greats who ever comes to visit it.
    """
    string = "[MIN 5 [MAX 30 900 ] ]"
    probs = [[0.1, 0.2, 0.3, 0.2, 0.15, 0.05],
             [0.1, 0.2, 0.4, 0.2, 0.1],
             [0.3, 0.4, 0.1, 0.2],
             [0.5, 0.1, 0.4],
             [0.6, 0.4],
             [1.0]]
    return render_template('hello.html', sample=string, prob_stack = probs)

if __name__ == '__main__':
    app.run()
