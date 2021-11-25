from typing import Iterable, Optional, Union

import torch

import bindsnet.network

class HomestaticLIFNodes(bindsnet.network.nodes.LIFNodes):
	"""
	Layer of `leaky integrate-and-fire (LIF) neurons that store a long time constant firing rate trace (for homeostatic plasticity)
	"""

	def __init__(
		self,
		n: Optional[int] = None,
		shape: Optional[Iterable[int]] = None,
		traces: bool = False,
		traces_additive: bool = False,
		tc_trace: Union[float, torch.Tensor] = 20.0,
		trace_scale: Union[float, torch.Tensor] = 1.0,
		homeostatic_traces: bool = True,
		tc_homeostatic_trace: Union[float, torch.Tensor] = 1000.0,
		sum_input: bool = False,
		thresh: Union[float, torch.Tensor] = -52.0,
		rest: Union[float, torch.Tensor] = -65.0,
		reset: Union[float, torch.Tensor] = -65.0,
		refrac: Union[int, torch.Tensor] = 5,
		tc_decay: Union[float, torch.Tensor] = 100.0,
		lbound: float = None,
		**kwargs,
	) -> None:
		"""
		Instantiates a layer of LIF neurons.
		:param n: The number of neurons in the layer.
		:param shape: The dimensionality of the layer.
		:param traces: Whether to record spike traces.
		:param traces_additive: Whether to record spike traces additively.
		:param tc_trace: Time constant of spike trace decay.
		:param trace_scale: Scaling factor for spike trace.
		:param homeostatic_traces: Whether to record a homeostatic firing rate trace.
		:param tc_homeostatic_trace: Time constant of the homeostatic firing rate trace decay.
		:param sum_input: Whether to sum all inputs.
		:param thresh: Spike threshold voltage.
		:param rest: Resting membrane voltage.
		:param reset: Post-spike reset voltage.
		:param refrac: Refractory (non-firing) period of the neuron.
		:param tc_decay: Time constant of neuron voltage decay.
		:param lbound: Lower bound of the voltage.
		"""
		super().__init__(
			n=n,
			shape=shape,
			traces=traces,
			traces_additive=traces_additive,
			tc_trace=tc_trace,
			trace_scale=trace_scale,
			sum_input=sum_input,
			thresh=thresh,
			rest=rest,
			reset=reset,
			refrac=refrac,
			tc_decay=tc_decay,
			lbound=None,
			**kwargs,
		)

		self.homeostatic_traces = homeostatic_traces
		self.tc_homeostatic_trace = torch.tensor(tc_homeostatic_trace)
		if self.homeostatic_traces:
			self.register_buffer("r", torch.Tensor())
			self.register_buffer("homeostatic_decay", torch.FloatTensor())

	def forward(self, x: torch.Tensor) -> None:
		"""
		Runs a single simulation step.
		:param x: Inputs to the layer.
		"""
		super().forward(x)

		if self.homeostatic_traces:
			self.r *= self.homeostatic_decay
			self.r += self.s.float()


	def reset_state_variables(self) -> None:
		"""
		Resets relevant state variables.
		"""
		super().reset_state_variables()
		self.r.zero_()

	def compute_decays(self, dt) -> None:
		"""
		Sets the relevant decays.
		"""
		super().compute_decays(dt=dt)
		self.homeostatic_decay = torch.exp(-torch.tensor(dt) / self.tc_homeostatic_trace) # Neuron homeostatic firing rate decay

	def set_batch_size(self, batch_size) -> None:
		"""
		Sets mini-batch size. Called when layer is added to a network.
		:param batch_size: Mini-batch size.
		"""
		super().set_batch_size(batch_size=batch_size)
		if self.homeostatic_traces:
			self.r = torch.zeros(batch_size, *self.shape, device=self.r.device)