"""Unit tests for wldstk_ml.utils.time_profiling."""

import logging
import re
import time
import unittest
from unittest import mock

from wldstk_ml.utils import time_profiling
from wldstk_ml.utils.logger import get_logger
from wldstk_ml.utils.time_profiling import log_time

LOGGER_NAME = "wldstk_ml.tests.time_profiling"
DEFAULT_LOGGER_NAME = "wldstk_ml.utils.time_profiling"


def fake_clocks(wall: float = 0.0, cpu: float = 0.0) -> mock.MagicMock:
    """Return a stand-in for the time module that reports the given durations."""
    clocks = mock.MagicMock()
    clocks.perf_counter.side_effect = [100.0, 100.0 + wall]
    clocks.process_time.side_effect = [50.0, 50.0 + cpu]
    return clocks


class TestLogTimeMessage(unittest.TestCase):
    """Tests the content of the log message, using fake clocks."""

    def setUp(self) -> None:
        """Create the logger that receives the messages."""
        self.logger = get_logger(LOGGER_NAME)

    def assert_message(self, expected: str, wall: float = 0.0, cpu: float = 0.0) -> None:
        """Assert that a block with the given durations logs the expected message."""
        with mock.patch.object(time_profiling, "time", fake_clocks(wall, cpu)):
            with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
                with log_time("Step", self.logger):
                    pass
        self.assertEqual([r.getMessage() for r in logs.records], [expected])

    def test_logs_label_wall_and_cpu_time(self) -> None:
        """Wall-clock and CPU time are measured separately."""
        self.assert_message("Step took 250.000 ms (CPU: 125.000 ms)", wall=0.25, cpu=0.125)

    def test_cpu_time_can_exceed_wall_time(self) -> None:
        """Multithreaded code uses more CPU time than wall-clock time."""
        self.assert_message("Step took 500.000 ms (CPU: 2.000 s)", wall=0.5, cpu=2.0)

    def test_duration_units(self) -> None:
        """Durations below one second use ms, the others use s."""
        cases = (
            (0.0, "0.000 ms"),
            (0.0015, "1.500 ms"),
            (0.9995, "999.500 ms"),
            (1.0, "1.000 s"),
            (75.5, "75.500 s"),
        )
        for seconds, text in cases:
            with self.subTest(seconds=seconds):
                self.assert_message(f"Step took {text} (CPU: 0.000 ms)", wall=seconds)


class TestLogTimeLogger(unittest.TestCase):
    """Tests which logger and level log_time writes to."""

    def setUp(self) -> None:
        """Create the logger that receives the messages."""
        self.logger = get_logger(LOGGER_NAME)

    def test_default_label(self) -> None:
        """Without a label the message starts with 'Block'."""
        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            with log_time(logger=self.logger):
                pass
        self.assertTrue(logs.records[0].getMessage().startswith("Block took "))

    def test_default_logger(self) -> None:
        """Without a logger the module's own logger is used."""
        with self.assertLogs(DEFAULT_LOGGER_NAME, logging.INFO) as logs:
            with log_time("Step"):
                pass
        self.assertEqual(logs.records[0].name, DEFAULT_LOGGER_NAME)

    def test_custom_logger_replaces_default_logger(self) -> None:
        """A given logger receives the message instead of the default one."""
        with self.assertNoLogs(DEFAULT_LOGGER_NAME, logging.DEBUG):
            with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
                with log_time("Step", self.logger):
                    pass
        self.assertEqual(logs.records[0].name, LOGGER_NAME)

    def test_default_level_is_info(self) -> None:
        """Messages are logged at INFO unless requested otherwise."""
        with self.assertLogs(LOGGER_NAME, logging.DEBUG) as logs:
            with log_time("Step", self.logger):
                pass
        self.assertEqual(logs.records[0].levelno, logging.INFO)

    def test_custom_level(self) -> None:
        """The level argument sets the level of the message."""
        with self.assertLogs(LOGGER_NAME, logging.DEBUG) as logs:
            with log_time("Step", self.logger, logging.DEBUG):
                pass
        self.assertEqual(logs.records[0].levelno, logging.DEBUG)

    def test_message_below_logger_level_is_dropped(self) -> None:
        """The logger's level filter still applies."""
        with self.assertNoLogs(LOGGER_NAME, logging.INFO):
            with log_time("Step", self.logger, logging.DEBUG):
                pass


class TestLogTimeBehavior(unittest.TestCase):
    """Tests log_time around real code."""

    def setUp(self) -> None:
        """Create the logger that receives the messages."""
        self.logger = get_logger(LOGGER_NAME)

    def test_exception_is_logged_and_reraised(self) -> None:
        """A failing block is still timed, and the exception is not swallowed."""
        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            with self.assertRaisesRegex(ValueError, "boom"):
                with log_time("Step", self.logger):
                    raise ValueError("boom")
        self.assertEqual(len(logs.records), 1)

    def test_decorator_keeps_function_behavior(self) -> None:
        """A decorated function keeps its arguments, result and name."""

        @log_time("Double", self.logger)
        def double(value: int, factor: int = 2) -> int:
            return value * factor

        with self.assertLogs(LOGGER_NAME, logging.INFO):
            self.assertEqual(double(3, factor=4), 12)
        self.assertEqual(double.__name__, "double")

    def test_decorator_logs_every_call(self) -> None:
        """Each call of a decorated function is timed separately."""

        @log_time("Call", self.logger)
        def call() -> None:
            pass

        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            call()
            call()
        self.assertEqual(len(logs.records), 2)

    def test_decorator_supports_recursion(self) -> None:
        """Nested calls of one decorated function do not share timers."""

        @log_time("Countdown", self.logger)
        def countdown(value: int) -> int:
            return 0 if value == 0 else countdown(value - 1)

        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            countdown(2)
        self.assertEqual(len(logs.records), 3)

    def test_nested_blocks_log_inner_block_first(self) -> None:
        """Each block reports when it ends."""
        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            with log_time("Outer", self.logger):
                with log_time("Inner", self.logger):
                    pass
        labels = [r.getMessage().split()[0] for r in logs.records]
        self.assertEqual(labels, ["Inner", "Outer"])

    def test_measures_real_time(self) -> None:
        """Without fake clocks, the reported wall-clock time covers the block."""
        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            with log_time("Sleep", self.logger):
                time.sleep(0.02)
        match = re.fullmatch(
            r"Sleep took ([\d.]+) ms \(CPU: [\d.]+ m?s\)", logs.records[0].getMessage()
        )
        assert match is not None
        self.assertGreaterEqual(float(match.group(1)), 19.0)


if __name__ == "__main__":
    unittest.main()
