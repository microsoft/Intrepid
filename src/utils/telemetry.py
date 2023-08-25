import time


class Telemeter:
    """A class for measuring time taken by different code snippets"""

    def __init__(self, name):
        self.name = name

        self.time_data = dict()
        self.ctr = dict()

        self.active_key = None
        self.timer = None  # Start of current code window

    def start(self, key):
        self.active_key = "%s_%s" % (self.name, key)

        # Start the timer
        self.timer = time.time()

    def stop(self):
        time_taken = time.time() - self.timer
        self.timer = None  # Reset the timer

        if self.active_key in self.time_data:
            self.time_data[self.active_key] += time_taken
            self.ctr[self.active_key] += 1
        else:
            self.time_data[self.active_key] = time_taken
            self.ctr[self.active_key] = 1

    def merge(self, telemetry):
        assert self.timer is not None and telemetry is not None, "Cannot merge telemeters with running timer."
        assert self.name != telemetry.name, "Telemeters with same name cannot be merged."

        for key in telemetry.time_data:
            self.time_data[key] = telemetry.time_data[key]
            self.ctr[key] = telemetry.ctr[key]

    def save_to_log(self, logger):
        for key, time_taken in sorted(self.time_data.items()):
            count = self.ctr[key]
            avg = round(time_taken / float(max(1, count)), 4)
            logger.log("%r: Avg time taken %r sec with %d count" % (key, avg, count))

    def print_report(self):
        for key, time_taken in sorted(self.time_data.items()):
            count = self.ctr[key]
            avg = round(time_taken / float(max(1, count)), 4)
            print("%r: Avg time taken %r sec with %d count" % (key, avg, count))
