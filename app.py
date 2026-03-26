from flask import Flask

app = Flask(__name__)

from routes.DOGCAT import predict_bp
app.register_blueprint(predict_bp)

if __name__ == '__main__':
    app.run(debug=True)
