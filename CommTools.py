from time import sleep
from functools import cache
from collections.abc import Iterable


class EipComm:  # EthernetIP communication
    def __init__(self, ip):
        from pycomm3 import LogixDriver
        self.ip = ip
        self.plc = LogixDriver(ip)
        self.plc.open()
        assert self.plc.connected, f"Could not connect to PLC on {ip}"
        self.running_monitors = {}

    def close(self):
        self.plc.close()

    def read(self, tag):
        return self.plc.read(tag).value

    def read_multiples(self, tags):
        return [t.value for t in self.plc.read(*tags)]

    def write(self, tag, value):
        self.plc.write((tag, value))

    def write_multiples(self, tags, values):
        messages = [(tag, value) for tag, value in zip(tags, values)]
        self.plc.write(*messages)

    def stop_monitor(self, tag):
        monitor = self.running_monitors.pop(tag)
        monitor.timer.stop()

    def stop_monitor_all(self):
        for tag in self.running_monitors.keys():
            self.stop_monitor(tag)

    def monitor_tag(self, tag, python_class, interval_ms):
        from .QtTools import TagMonitor
        self.running_monitors[tag] = monitor = TagMonitor(tag, self, python_class, interval_ms)
        return monitor.change_signal

    def await_value(self, tag, continue_value, interval_ms=50):
        while self.read(tag) != continue_value:
            sleep(interval_ms / 1000)


class TurckIoComm:  # Communication to turck input/output module.
    def __init__(self, ip):
        from pycomm3 import CIPDriver
        self.ip = ip
        self.plc = CIPDriver(ip)
        assert self.plc.open(), f"Could not connect to PLC on {ip}"
        self.model = self.identify_model()
        self.read_input_parameters = {'TBEN-S1-8DIP': {'class_code': 148, 'instance': 1, 'attribute_offset': 3}}

    def identify_model(self):
        from pycomm3 import Services, ClassCode, ModuleIdentityObject
        response = self.plc.generic_message(
            service=Services.get_attributes_all,
            class_code=ClassCode.identity_object,
            instance=b"\x01",
            data_type=ModuleIdentityObject)
        return response.value['product_name']

    def close(self):
        self.plc.close()

    def read_input(self, channel, class_code=None, instance=None, attribute_offset=None):
        # Parameters from documentation at www.turck.de/attachment/100001931.pdf. Page 147 and forward
        from pycomm3 import Services, DataTypes

        pars = self.read_input_parameters.get(self.model)
        class_code = class_code or pars['class_code']
        instance = instance or pars['instance']
        attribute_offset = attribute_offset or pars['attribute_offset']

        response = self.plc.generic_message(
            service=Services.get_attribute_single,
            class_code=class_code,
            instance=instance,
            attribute=attribute_offset + channel,
            data_type=DataTypes.usint)  #connected=True/False  # Not sure if this could be needed
        return bool(response.value)

class OpcuaComm:
    def __init__(self, ip, port=4840):
        from opcua import Client  # pip install opcua
        from opcua.ua import VariantType as types
        self.client = Client(f"opc.tcp://{ip}:{port}")
        self.client.connect()
        self.opcua_types = {bool: types.Boolean, int: types.Int32, float: types.Float, str: types.ByteString}
        self.wait_time_ms = 50 / 1000

    def close(self):
        self.client.disconnect()

    @cache
    def get_node(self, node_id):
        return self.client.get_node(node_id)

    def package_value(self, value, type):
        from opcua import ua
        assert type is not None or type in self.opcua_types, "OPCUA cannot guess type, please specify it."
        type = type or self.opcua_types[type(value)]
        variant_value = ua.Variant(value, type)
        return variant_value

    def read(self, node_id, wait_till_true=False):
        node = self.get_node(node_id)
        value = node.get_value()
        while not value and wait_till_true:
            sleep(self.wait_time_ms)
            value = node.get_value()
        return value

    def read_multiples(self, node_ids):
        nodes = [self.get_node(id) for id in node_ids]
        values = self.client.get_values(nodes)
        return values

    def write(self, node_id, value, type=None):
        node = self.get_node(node_id)
        node.set_value(self.package_value(value, type))

    def write_multiples(self, node_ids, values, types):
        if not isinstance(types, Iterable):
            types = len(values) * [types]
        variant_values = [self.package_value(v, t) for v, t in zip(values, types)]
        nodes = [self.get_node(id) for id in node_ids]
        self.client.set_values(nodes, variant_values)
