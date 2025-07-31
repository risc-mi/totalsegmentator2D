import sys

def _default_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


_log_methods = [_default_print]
_logged_contexts = set()


def log_silent(silent: bool):
    global _log_methods
    _log_methods = [] if silent else [_default_print]

def log(*args, **kwargs):
    if 'once' in kwargs:
        if kwargs.get('once', False):
            global _logged_contexts
            import inspect
            ctx = inspect.stack()
            ctx = '\n'.join('{}:{}'.format(c.filename, c.lineno) for c in ctx)
            if ctx in _logged_contexts:
                return
            _logged_contexts.add(ctx)
        del kwargs['once']

    global _log_methods
    for _log in _log_methods:
        _log(*args, **kwargs)

def warn(msg, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    log(msg, **kwargs)
    sys.stderr.flush()
    sys.stdout.flush()
