#!/usr/bin/env python3

from multiverse_client_py import MultiverseClient, MultiverseMetaData

class MultiverseSmoother(MultiverseClient):
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

import argparse
import time
import numpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Smoothing data from multiverse.")

    # Define arguments
    parser.add_argument("--world_name", type=str, required=False, default="world", help="Name of the world")
    parser.add_argument("--simulation_name", type=str, required=False, default="multiverse_smoothing", help="Name of the simulation")
    parser.add_argument("--port", type=str, required=False, default="7554", help="Port number")
    parser.add_argument("--object_names", type=str, required=False, default="", help="Object names to smooth data from")
    parser.add_argument("--attribute_names", type=str, required=False, default="", help="Attribute names to smooth data from")
    parser.add_argument("--map", type=str, required=False, default="joint,actuator", help="Map from data to smooth data")
    parser.add_argument("--time_constant", type=float, default=0.02, required=False, help="Time constant for smoothing")

    # Parse arguments
    args = parser.parse_args()

    receive_data = []

    multiverse_meta_data = MultiverseMetaData(
        world_name=args.world_name,
        simulation_name=args.simulation_name,
        length_unit="m",
        angle_unit="rad",
        mass_unit="kg",
        time_unit="s",
        handedness="rhs",
    )
    multiverse_logger = MultiverseSmoother(port=args.port, multiverse_meta_data=multiverse_meta_data)
    multiverse_logger.run()

    object_names = args.object_names.split(",")
    attribute_names = args.attribute_names.split(",")
    map_data = args.map.split(",")

    if attribute_names == [""]:
        multiverse_logger.request_meta_data["send"] = {}
        multiverse_logger.request_meta_data["receive"] = {}
        for object_name in object_names:
            multiverse_logger.request_meta_data["receive"][object_name] = [""]
        while True:
            multiverse_logger.send_and_receive_meta_data()
            if "receive" in multiverse_logger.response_meta_data:
                for object_name in object_names:
                    if object_name not in multiverse_logger.response_meta_data["receive"]:
                        break
                else:
                    break
            time.sleep(1.0)
        multiverse_logger.request_meta_data["send"] = {}
        for object_name, object_attributes in multiverse_logger.response_meta_data["receive"].items():
            send_object_name = object_name.replace(map_data[0],map_data[1])
            multiverse_logger.request_meta_data["send"][f"{send_object_name}"] = []
            for attribute_name in object_attributes.keys():
                if attribute_name in ["joint_rvalue", "joint_tvalue", "joint_angular_velocity", "joint_linear_velocity"]:
                    multiverse_logger.request_meta_data["send"][send_object_name].append(f"cmd_{attribute_name}")
                else:
                    raise ValueError(f"Unknown attribute name: {attribute_name}")
    else:
        multiverse_logger.request_meta_data["send"] = {}
        multiverse_logger.request_meta_data["receive"] = {}
        for object_name in object_names:
            if any(attribute_name not in ["joint_rvalue", "joint_tvalue", "joint_angular_velocity", "joint_linear_velocity"] for attribute_name in attribute_names):
                raise ValueError(f"Unknown attribute name in {object_name}: {attribute_names}")
            multiverse_logger.request_meta_data["send"][f"{object_name.replace(map_data[0],map_data[1])}"] = [f"cmd_{attribute_name}" for attribute_name in attribute_names]
            multiverse_logger.request_meta_data["receive"][object_name] = attribute_names

    try:
        while True:
            multiverse_logger.send_and_receive_meta_data()
            if "receive" in multiverse_logger.response_meta_data:
                for object_name in object_names:
                    if object_name not in multiverse_logger.response_meta_data["receive"]:
                        break
                else:
                    break
            time.sleep(1.0)
        while True:
            send_data = []
            for object_name, object_data in multiverse_logger.response_meta_data["receive"].items():
                for attribute_name, attribute_values in object_data.items():
                    send_data += attribute_values
            if any(x is None for x in send_data):
                time.sleep(1.0)
            else:
                break
    except KeyboardInterrupt:
        multiverse_logger.stop()
        exit(0)

    send_data = numpy.array(send_data)
    multiverse_logger.send_data = [0.0] + send_data.tolist()
    multiverse_logger.send_and_receive_data()

    T = args.time_constant
    D = 1
    T1 = T * T
    T2 = 2 * D * T

    y_1 = numpy.array(send_data)
    y_2 = numpy.array(send_data)

    starting_time = multiverse_logger.sim_time
    current_time = starting_time
    t_2 = current_time

    multiverse_logger.send_data = [current_time] + send_data.tolist()
    multiverse_logger.send_and_receive_data()
    current_time = multiverse_logger.sim_time
    t_1 = current_time

    current_step = current_time
    try:
        while multiverse_logger.receive_data[0] >= 0.0:
            current_time = multiverse_logger.sim_time
            delta_T_1 = current_time - t_1
            delta_T_2 = t_1 - t_2
            u = numpy.array(multiverse_logger.receive_data[1:])

            if current_time - current_step >= 0.001:
                send_data = (delta_T_1 * delta_T_2 / T1) * (u
                                                            + (-T2 / delta_T_1 + T1 / delta_T_1 ** 2 + T1 / (delta_T_1 * delta_T_2)) * y_1
                                                            + (-1 + T2 / delta_T_1 - T1 / delta_T_1 ** 2) * y_2)
                current_step = current_time

            multiverse_logger.send_data = [current_time] + send_data.tolist()
            multiverse_logger.send_and_receive_data()

            t_2 = t_1
            t_1 = current_time
            y_2 = y_1
            y_1 = send_data
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass

    multiverse_logger.stop()
