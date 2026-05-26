class LinearSchedule:
	"""Linear ramp from `initial` to `final` over `ramp_steps` advances, then hold at `final`."""

	def __init__(self, initial: float, final: float, ramp_steps: int):
		self.initial = initial
		self.final = final
		self.ramp = max(1, ramp_steps)
		self._n = 0

	@property
	def value(self) -> float:
		frac = min(1.0, self._n / self.ramp)
		return self.initial + (self.final - self.initial) * frac

	def step(self, n: int = 1) -> None:
		self._n += n
