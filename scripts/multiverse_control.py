#!/usr/bin/env python3

from multiverse_client_py import MultiverseClient, MultiverseMetaData

import argparse
import time
import os
import yaml
import dataclasses
from typing import Tuple, Optional, List
import numpy


@dataclasses.dataclass
class Gain:
    kp: float = 1.0
    kv: Optional[float] = None

    def calculate(self, value: numpy.ndarray) -> numpy.ndarray:
        assert self.kv is None, "Velocity gain (kv) is not supported in this implementation yet."
        return self.kp * value


objects = {"send": {}, "receive": {}}


@dataclasses.dataclass
class MultiverseData:
    gain: Gain = dataclasses.field(default_factory=lambda: Gain(kp=1.0, kv=None))
    range: Optional[Tuple[float, float]] = None
    depends_on: Optional[List[str]] = None
    _value: Optional[numpy.ndarray] = None

    def set_value(self, value: numpy.ndarray) -> None:
        if self._value is not None:
            assert value.shape == self.value.shape, f"Value shape {value.shape} does not match expected shape {self.value.shape}."
        dependency_sum = numpy.zeros_like(value)
        if self.depends_on is not None:
            for dependency in self.depends_on:
                assert dependency in objects["receive"], f"Dependency {dependency} not found in receive objects."
                for dependency_value in objects["receive"][dependency].values():
                    dependency_sum += dependency_value.value
        self._value = self.gain.calculate(value + dependency_sum)
        if self.range is not None:
            self._value = numpy.clip(self._value, self.range[0], self.range[1])

    def compute_value(self) -> numpy.ndarray:
        self.set_value(numpy.zeros_like(self.value))
        return self.value

    @property
    def value(self) -> numpy.ndarray:
        assert self._value is not None, f"Value has not been set."
        return self._value


class MultiverseController(MultiverseClient):
    def __init__(self, port: str, multiverse_meta_data: MultiverseMetaData) -> None:
        super().__init__(port, multiverse_meta_data)

    def loginfo(self, message: str) -> None:
        print(f"INFO: {message}")

    def logwarn(self, message: str) -> None:
        print(f"WARN: {message}")

    def _run(self) -> None:
        self.loginfo("Start smoothing.")
        self._connect_and_start()

    def send_and_receive_meta_data(self) -> None:
        self.loginfo("Sending request meta data: " + str(self.request_meta_data))
        self._communicate(True)
        self.loginfo("Received response meta data: " + str(self.response_meta_data))

    def send_and_receive_data(self) -> None:
        self._communicate(False)

    def initialize_objects(self) -> bool:
        self.send_and_receive_meta_data()
        response_meta_data = self.response_meta_data
        for send_receive in ["receive", "send"]:
            if send_receive not in response_meta_data:
                return False
            for object_name in objects[send_receive].keys():
                if object_name not in response_meta_data[send_receive]:
                    return False
                for attribute_name, attribute_values in response_meta_data[send_receive][object_name].items():
                    assert attribute_name in objects[send_receive][object_name], f"Attribute {attribute_name} not found in object {object_name}."
                    if any([attribute_value is None for attribute_value in attribute_values]):
                        return False
                    objects[send_receive][object_name][attribute_name].set_value(numpy.array(attribute_values))
        return True

    def loop(self) -> None:
        send_data = []
        response_meta_data = self.response_meta_data
        for object_name, object_data in response_meta_data["send"].items():
            assert object_name in objects["send"], f"Object {object_name} not found in send objects."
            for attribute_name in object_data.keys():
                assert attribute_name in objects["send"][object_name], f"Attribute {attribute_name} not found in object {object_name}."
                send_data += objects["send"][object_name][attribute_name].compute_value().tolist()
        self.send_data = [self.sim_time] + send_data
        self.send_and_receive_data()
        receive_data = self.receive_data[1:]
        for object_name, object_data in response_meta_data["receive"].items():
            assert object_name in objects["receive"], f"Object {object_name} not found in receive objects."
            for attribute_name in object_data.keys():
                assert attribute_name in objects["receive"][object_name], f"Attribute {attribute_name} not found in object {object_name}."
                objects["receive"][object_name][attribute_name].set_value(numpy.array([receive_data.pop(0)]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Control data from multiverse.")

    # Define arguments
    parser.add_argument(
        "--world_name",
        type=str,
        required=False,
        default="world",
        help="Name of the world",
    )
    parser.add_argument(
        "--simulation_name",
        type=str,
        required=False,
        default="multiverse_smoothing",
        help="Name of the simulation",
    )
    parser.add_argument("--port", type=str, required=False, default="7555", help="Port number")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to load the configuration file",
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=False,
        default=1000.0,
        help="Rate at which to run the control loop (default 1000Hz)",
    )

    # Parse arguments
    args = parser.parse_args()

    config_path = args.data_path
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist."

    # Load configuration
    config = yaml.safe_load(open(config_path, "r"))

    multiverse_meta_data = MultiverseMetaData(
        world_name=args.world_name,
        simulation_name=args.simulation_name,
        length_unit="m",
        angle_unit="rad",
        mass_unit="kg",
        time_unit="s",
        handedness="rhs",
    )
    multiverse_controller = MultiverseController(port=args.port, multiverse_meta_data=multiverse_meta_data)
    multiverse_controller.run()

    multiverse_controller.request_meta_data["send"] = {}
    multiverse_controller.request_meta_data["receive"] = {}
    for output_object, output_data in config.items():
        multiverse_controller.request_meta_data["send"][output_object] = []
        objects["send"][output_object] = {}
        for output_attribute, output_attribute_data in output_data.items():
            multiverse_controller.request_meta_data["send"][output_object].append(output_attribute)
            dependencies = []
            for input_object, dependency_data in output_attribute_data.get("depends_on", {}).items():
                if input_object not in objects["receive"]:
                    objects["receive"][input_object] = {}
                if input_object not in multiverse_controller.request_meta_data["receive"]:
                    multiverse_controller.request_meta_data["receive"][input_object] = []
                for input_attribute, input_attribute_data in dependency_data.items():
                    multiverse_controller.request_meta_data["receive"][input_object].append(input_attribute)
                    gain_dict = input_attribute_data.get("gain", None)
                    gain = Gain(kp=gain_dict.get("kp", 1.0), kv=gain_dict.get("kv", None)) if gain_dict is not None else Gain()
                    objects["receive"][input_object][input_attribute] = MultiverseData(
                        gain=gain,
                        range=input_attribute_data.get("range", None),
                    )
                    dependencies.append(input_object)
            gain_dict = output_attribute_data.get("gain", None)
            gain = Gain(kp=gain_dict.get("kp", 1.0), kv=gain_dict.get("kv", None)) if gain_dict is not None else Gain()
            objects["send"][output_object][output_attribute] = MultiverseData(
                gain=gain, range=output_attribute_data.get("range", None), depends_on=dependencies
            )

    try:
        while not multiverse_controller.initialize_objects():
            print("Waiting for data to be ready...")
            time.sleep(1.0)
        while True:
            start_time = time.time()
            multiverse_controller.loop()
            sleep_time = max(0.0, (1.0 / args.rate) - time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        multiverse_controller.stop()
        exit(0)

    multiverse_controller.stop()
