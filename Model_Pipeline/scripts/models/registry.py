"""Builds models from the config: family name to class.

Owner: P2.

The config names a family for each target (or one family for several). This
module turns that into one model, or a composite of several, and refuses an
unknown family or a target no model covers.
"""
