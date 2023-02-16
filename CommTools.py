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
