"""Compatibility facade for the canonical Podman subprocess runner."""

from alphaapollo.common.execution.backends._compat import alias_module as _alias_module

_alias_module(__name__, "alphaapollo.common.execution.sandbox._podman.runner")
# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
