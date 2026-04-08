from ._interpolator_selector import _build_interpolator

class Interpolator:
    def __init__(self, coordinates, values, **kwargs):
        self._implementation = _build_interpolator(coordinates, values, **kwargs)

        # Direct binding for better performance
        self.interpolatoe = self._implementation.interpolate
