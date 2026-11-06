"""setup_logging must work whether or not logging was configured before YAIB starts (#260)."""

import logging

import pytest

from icu_benchmarks.run_utils import setup_logging

LOGGERS = ["", "pytorch_lightning", "lightning_fabric"]
LOG_FORMAT, DATE_FORMAT = "%(levelname)s|%(message)s", "%H"


@pytest.fixture
def restore_loggers():
    # setup_logging reformats existing handlers (such as pytest's), so their formatters are restored too
    saved = {
        name: (logging.getLogger(name).level, [(h, h.formatter) for h in logging.getLogger(name).handlers]) for name in LOGGERS
    }
    yield
    for name, (level, handlers) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers[:] = [handler for handler, _ in handlers]
        for handler, formatter in handlers:
            handler.setFormatter(formatter)


def test_lightning_loggers_without_handlers(restore_loggers):
    # Lightning adds no handler when the root logger already had one (pytest, notebooks, logging.basicConfig)
    for name in LOGGERS[1:]:
        logging.getLogger(name).handlers.clear()
    setup_logging(DATE_FORMAT, LOG_FORMAT, verbose=False)
    assert logging.getLogger().level == logging.INFO
    assert all(logging.getLogger(name).level == logging.INFO for name in LOGGERS[1:])


@pytest.mark.parametrize("verbose", [False, True])
def test_lightning_handlers_get_the_format(restore_loggers, verbose):
    handlers = {name: logging.StreamHandler() for name in LOGGERS[1:]}
    for name, handler in handlers.items():
        logging.getLogger(name).handlers[:] = [handler, logging.StreamHandler()]
    setup_logging(DATE_FORMAT, LOG_FORMAT, verbose=verbose)
    level = logging.DEBUG if verbose else logging.INFO
    for name in LOGGERS[1:]:
        logger = logging.getLogger(name)
        assert logger.level == level
        assert all(h.formatter._fmt == LOG_FORMAT and h.formatter.datefmt == DATE_FORMAT for h in logger.handlers)


def test_existing_root_handlers_get_the_format(restore_loggers):
    # basicConfig does not touch a root logger that already has handlers, so they are formatted in place
    handler = logging.StreamHandler()
    logging.getLogger().handlers[:] = [handler]
    setup_logging(DATE_FORMAT, LOG_FORMAT, verbose=False)
    assert logging.getLogger().handlers == [handler]
    assert handler.formatter._fmt == LOG_FORMAT and handler.formatter.datefmt == DATE_FORMAT
