# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Smoke tests for the rivoli package."""

import rivoli


def test_version_is_resolved():
    assert rivoli.__version__ != "0+unknown"
