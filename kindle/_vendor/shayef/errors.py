"""Error types. Every message is written to be read by someone in a notebook
cell, so they say what to do next rather than only what went wrong."""


class ShayefError(Exception):
    """Base class for every error shayef raises on purpose."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self):
        if self.hint:
            return "%s\n  -> %s" % (self.message, self.hint)
        return self.message


class ConfigError(ShayefError):
    pass


class ControlPlaneError(ShayefError):
    """shayef.me refused or could not answer a request."""

    def __init__(self, message, status=None, hint=None):
        super().__init__(message, hint=hint)
        self.status = status


class RelayError(ShayefError):
    """Something went wrong fetching or running the relay connector."""


class TunnelError(ShayefError):
    pass
