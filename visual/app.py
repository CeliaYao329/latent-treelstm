from flask import Flask, render_template, request
from listops.reinforce.train_reinforce_model import main
import argparse

global sample_string
global probs
# App config.
app = Flask(__name__)


@app.route('/_stuff', methods= ['GET'])
def update_visual(sample, probs):
    print("-------------------------------------------")
    return render_template('hello.html', sample=sample, prob_stack=probs)


@app.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return render_template('index.html')


@app.route('/vis/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    args = {"word-dim": 128,
            "hidden-dim": 128,
            "parser-leaf-transformation": "lstm_transformation",
            "parser-trans-hidden_dim": 128,
            "tree-leaf-transformation": "no_transformation",
            "tree-trans-hidden_dim": 128,
            "baseline-type": "self_critical",
            "var-normalization": "True",
            "entropy-weight": 0.0001,
            "clip-grad-norm": 0.5,
            "optimizer": "adadelta",
            "env-lr": 1.0,
            "pol-lr": 1.0,
            "lr-scheduler-patience": 8,
            "l2-weight": 0.0001,
            "batch-size": 4,
            "max-epoch": 300,
            "es-patience": 20,
            "es-threshold": 0.005,
            # Test
            "gpu-id": -1,
            "model-dir": "data/listops/reinforce/models/exp0",
            "logs-path": "data/listops/reinforce/logs/exp0",
            "tensorboard-path": "data/listops/reinforce/tensorboard/exp0",
            "vis_sample": text,
            }
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-dim", required=False, default=args["word-dim"], type=int)
    parser.add_argument("--hidden-dim", required=False, default=args["hidden-dim"], type=int)
    parser.add_argument("--parser-leaf-transformation", required=False, default=args["parser-leaf-transformation"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--parser-trans-hidden_dim", required=False, default=args["parser-trans-hidden_dim"], type=int)
    parser.add_argument("--tree-leaf-transformation", required=False, default=args["tree-leaf-transformation"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--tree-trans-hidden_dim", required=False, default=args["tree-trans-hidden_dim"], type=int)

    parser.add_argument("--baseline-type", default=args["baseline-type"],
                        choices=["no_baseline", "ema", "self_critical"])
    parser.add_argument("--var-normalization", default=args["var-normalization"],
                        type=lambda string: True if string == "True" else False)
    parser.add_argument("--entropy-weight", default=args["entropy-weight"], type=float)
    parser.add_argument("--clip-grad-norm", default=args["clip-grad-norm"], type=float,
                        help="If the value is less or equal to zero clipping is not performed.")

    parser.add_argument("--optimizer", required=False, default=args["optimizer"], choices=["adam", "sgd", "adadelta"])
    parser.add_argument("--env-lr", required=False, default=args["env-lr"], type=float)
    parser.add_argument("--pol-lr", required=False, default=args["pol-lr"], type=float)
    parser.add_argument("--lr-scheduler-patience", required=False, default=args["lr-scheduler-patience"], type=int)
    parser.add_argument("--l2-weight", required=False, default=args["l2-weight"], type=float)
    parser.add_argument("--batch-size", required=False, default=args["batch-size"], type=int)

    parser.add_argument("--max-epoch", required=False, default=args["max-epoch"], type=int)
    parser.add_argument("--es-patience", required=False, default=args["es-patience"], type=int)
    parser.add_argument("--es-threshold", required=False, default=args["es-threshold"], type=float)
    parser.add_argument("--gpu-id", required=False, default=args["gpu-id"], type=int)
    parser.add_argument("--model-dir", required=False, default=args["model-dir"], type=str)
    parser.add_argument("--logs-path", required=False, default=args["logs-path"], type=str)
    parser.add_argument("--tensorboard-path", required=False, default=args["tensorboard-path"], type=str)

    global_step = 0
    best_model_path = None
    args = parser.parse_args()
    args.vis_sample = text

    print(main(args))
    probs = [[0.1, 0.2, 0.3, 0.2, 0.15, 0.05],
             [0.1, 0.2, 0.4, 0.2, 0.1],
             [0.3, 0.4, 0.1, 0.2],
             [0.5, 0.1, 0.4],
             [0.6, 0.4],
             [1.0]]
    return render_template('hello.html', sample=text, prob_stack=probs)

@app.route('/vis/')
def hello():
    """ Displays the page greats who ever comes to visit it.
    """
    sample_string = "[MIN 5 [MAX 30 900 ] ]"
    probs = [[0.1, 0.2, 0.3, 0.2, 0.15, 0.05],
             [0.1, 0.2, 0.4, 0.2, 0.1],
             [0.3, 0.4, 0.1, 0.2],
             [0.5, 0.1, 0.4],
             [0.6, 0.4],
             [1.0]]
    return render_template('hello.html', sample=sample_string, prob_stack = probs)

if __name__ == '__main__':
    app.run()
