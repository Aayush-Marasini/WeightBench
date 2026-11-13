"""Construction calls use the normal engine's counters and append-only evidence."""
import pytest

from weightbench.harness.author import AuthorForwardMeter


def test_author_cap_and_failures_are_metered(tmp_path):
    meter = AuthorForwardMeter(tmp_path / 'private', 2, {'purpose': 'synthetic test'})
    assert meter.call(lambda: (3, 4)) == (3, 4)

    def fail():
        raise ValueError('invalid construction input')

    with pytest.raises(RuntimeError):
        meter.call(fail)
    called = []
    with pytest.raises(RuntimeError):
        meter.call(lambda: called.append(True))
    assert not called and meter.used == 2 and meter.ledger.verify()
    assert len(list(meter.ledger.entries('probe'))) == 2
    assert len(list(meter.ledger.entries('probe_rejected'))) == 1
    with pytest.raises(FileExistsError):
        AuthorForwardMeter(tmp_path / 'private', 2, {})


@pytest.mark.parametrize('cap', [-1, True, 1.5])
def test_invalid_author_cap_rejected(tmp_path, cap):
    with pytest.raises(ValueError):
        AuthorForwardMeter(tmp_path / 'private', cap, {})
