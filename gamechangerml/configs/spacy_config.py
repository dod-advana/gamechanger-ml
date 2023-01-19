from gamechangerml.src.utilities.borg import Borg


class SpacyConfig(Borg):
    def __init__(self):
        Borg.__init__(self)

    def _set_config(self, val):
        self._value = val

    def _get_config(self):
        return getattr(self, "_value", None)

    config = property(_get_config, _set_config)
