#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import traceback  # noqa: F401
import warnings
from copy import deepcopy
from pathlib import Path
from datetime import datetime  # noqa: F401

import yaml
from flask import request, jsonify, Flask, Response

from config import ServingConfig
from models import SeizurePredictionModel


seizure_app = Flask(__name__)

_ServingConfig = deepcopy(ServingConfig)

_LoadedModel = SeizurePredictionModel.from_file(_ServingConfig.model_path)


@seizure_app.route("/")
def hello_world():
    return "<p>Seizure Prediction APP</p>"


@seizure_app.route("/seizure_prediction", methods=["POST"])
def get_seizure_prediction() -> Response:
    """ """
    data = request.get_json(force=True)
    if data is None:
        # raise ValueError("No input data")
        result = {
            "code": 1,
            "error_type": "ValueError",
            "error_msg": "No input data",
        }
        return jsonify(result)
    if not isinstance(data, (dict, list)):
        # raise TypeError("Input data must be a dict or a list of dict")
        result = {
            "code": 2,
            "error_type": "TypeError",
            "error_msg": "Input data must be a dict or a list of dict",
        }
        return jsonify(result)
    try:
        result = _LoadedModel.pipeline(data)
        result = {
            "code": 0,
            "result": result,
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        result = {
            "code": 3,
            "error_type": error_type,
            "error_msg": error_msg,
        }
    return jsonify(result)


def parse_args() -> dict:
    """ """
    parser = argparse.ArgumentParser(
        description="Seizure Prediction APP",
    )
    parser.add_argument(
        "config_file_path",
        nargs=argparse.OPTIONAL,
        type=str,
        help="Config file (.yml or .yaml file) path",
    )
    parser.add_argument(
        "--ip",
        type=str,
        help="IP address",
        dest="ip",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11111,
        help="Port",
        dest="port",
    )

    args = vars(parser.parse_args())
    config_file_path = args.pop("config_file_path")
    if config_file_path is None:
        warnings.warn("No input config file path, use default config")
    else:
        config_file_path = Path(config_file_path).resolve()
        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file {config_file_path} not found")
        if config_file_path.suffix not in [".yml", ".yaml"]:
            raise ValueError(
                f"Config file {config_file_path} must be a .yml or .yaml file"
            )
        serving_config = yaml.safe_load(config_file_path.read_text())
        _ServingConfig.update(serving_config)

    return args


if __name__ == "__main__":
    # command for running in background
    # nohup python service.py --ip xx.xx.xx.xx [--port xxxx] > ./log/service.log 2>&1 & echo $! > ./log/service.pid
    args = parse_args()
    seizure_app.run(host=args["ip"], port=args["port"], debug=True)
