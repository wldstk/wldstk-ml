"""Unit tests for wldstk_ml.utils.logger."""

import io
import logging
import re
import unittest
from unittest import mock

from wldstk_ml.utils.logger import LOG_LEVEL, PACKAGE_LOGGER_NAME, get_logger


class CleanLoggerTestCase(unittest.TestCase):
    """Runs each test on an unconfigured package logger and restores it afterwards."""

    def setUp(self) -> None:
        """Remove the package logger's configuration for the duration of the test."""
        self.package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        state = (self.package_logger.handlers[:], self.package_logger.level)
        propagate = self.package_logger.propagate
        self.addCleanup(self._restore, state, propagate)
        self.package_logger.handlers.clear()
        self.package_logger.setLevel(logging.NOTSET)
        self.package_logger.propagate = True

    def _restore(self, state: tuple[list[logging.Handler], int], propagate: bool) -> None:
        """Put the original handlers, level and propagation back."""
        self.package_logger.handlers[:], level = state
        self.package_logger.setLevel(level)
        self.package_logger.propagate = propagate


class TestLoggerNames(CleanLoggerTestCase):
    """Tests how get_logger maps names into the package hierarchy."""

    def test_no_name_returns_package_logger(self) -> None:
        """None, an empty name and the package name all give the package logger."""
        for name in (None, "", PACKAGE_LOGGER_NAME):
            with self.subTest(name=name):
                self.assertIs(get_logger(name), self.package_logger)

    def test_package_module_name_is_kept(self) -> None:
        """Module names inside the package stay unchanged."""
        self.assertEqual(get_logger("wldstk_ml.utils.x").name, "wldstk_ml.utils.x")

    def test_outside_names_move_under_package(self) -> None:
        """Script names are placed under the package logger."""
        self.assertEqual(get_logger("__main__").name, "wldstk_ml.__main__")

    def test_similar_prefix_is_not_treated_as_package(self) -> None:
        """A name that only starts with the same letters is outside the package."""
        self.assertEqual(get_logger("wldstk_ml_other").name, "wldstk_ml.wldstk_ml_other")

    def test_same_name_returns_same_logger(self) -> None:
        """Repeated calls return one shared logger."""
        self.assertIs(get_logger("shared"), get_logger("shared"))


class TestLoggerConfiguration(CleanLoggerTestCase):
    """Tests the handler, level and propagation setup."""

    def test_level_is_debug(self) -> None:
        """The configured level is DEBUG, and it reaches child loggers."""
        self.assertEqual(LOG_LEVEL, logging.DEBUG)
        self.assertEqual(get_logger("child").getEffectiveLevel(), logging.DEBUG)

    def test_repeated_calls_add_one_handler(self) -> None:
        """Only one handler exists, however often get_logger is called."""
        for name in ("a", "b", "a", None):
            get_logger(name)
        self.assertEqual(len(self.package_logger.handlers), 1)
        self.assertIsInstance(self.package_logger.handlers[0], logging.StreamHandler)

    def test_child_loggers_have_no_own_handlers(self) -> None:
        """Child loggers rely on the package logger's handler."""
        self.assertEqual(get_logger("child").handlers, [])

    def test_does_not_propagate_to_root(self) -> None:
        """Messages stay out of the root logger, which prevents duplicates."""
        get_logger()
        self.assertFalse(self.package_logger.propagate)

    def test_root_logger_is_untouched(self) -> None:
        """Third-party loggers keep their defaults because root is not configured."""
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        get_logger("child")
        self.assertEqual(root.handlers, handlers)
        self.assertEqual(root.level, level)
        self.assertFalse(logging.getLogger("third_party").isEnabledFor(logging.DEBUG))


class TestLoggerOutput(CleanLoggerTestCase):
    """Tests what ends up on the console."""

    def test_writes_formatted_debug_message_to_stderr(self) -> None:
        """A DEBUG message appears once, with time, level, name and text."""
        with mock.patch("sys.stderr", new=io.StringIO()) as stderr:
            get_logger("__main__").debug("Loaded %d samples", 150)
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| DEBUG    \| "
            + re.escape("wldstk_ml.__main__ | Loaded 150 samples")
            + r"\n$",
        )

    def test_repeated_get_logger_does_not_duplicate_messages(self) -> None:
        """Several loggers in one process still write each message once."""
        with mock.patch("sys.stderr", new=io.StringIO()) as stderr:
            get_logger("a")
            get_logger("b").info("only once")
        self.assertEqual(stderr.getvalue().count("only once"), 1)


if __name__ == "__main__":
    unittest.main()
