from typing import Dict
from flask import Flask, render_template, request
from flask_cors import CORS

from openchat.envs import BaseEnv
from openchat.models import BaseModel
from werkzeug.utils import secure_filename
import openchat.config as cfg

questions = []


class WebDemoEnv(BaseEnv):
    """Base envrionment for web.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        CORS(self.app)

    def run(self, model: BaseModel):

        @self.app.route('/', methods=['GET', 'POST'])
        def index():

            return render_template('index.html', title=model.name)

        @self.app.route('/image', methods=['POST'])
        def save_image():
            f = request.files['file']
            print(f)
            fname = secure_filename(f.filename)
            f.save(cfg.image_path + '/' + fname)
            return {'output': 'ok! let\'s start'}

        @self.app.route('/send/<imageName>/<text>', methods=['GET'])
        def send(imageName, text: str) -> Dict[str, str]:

            if text in self.keywords:
                # Format of self.keywords dictionary
                # self.keywords['/exit'] = (exit_function, 'good bye.')

                # _out = self.keywords[text][1]
                # text to print when keyword triggered

                self.keywords[text][0](imageName, text)
                # function to operate when keyword triggered

            else:
                outputs = model.predict(imageName, text)

            return {'output': outputs}

        self.app.run(host='0.0.0.0', port=8080)
