"""Compatibility facade for :mod:`alphaapollo.common.execution.sandbox`."""

from alphaapollo.common.execution.backends._compat import forward_module as _forward_module

_forward_module(globals(), "alphaapollo.common.execution.sandbox")
# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
