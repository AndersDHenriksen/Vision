from time import sleep


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
