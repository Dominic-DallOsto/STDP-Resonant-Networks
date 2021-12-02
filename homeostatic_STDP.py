from typing import Optional, Sequence, Union
import torch

import bindsnet.learning

from bindsnet.network.topology import (
	AbstractConnection,
	Connection,
	LocalConnection,
)

class HomeostaticSTDP(bindsnet.learning.PostPre):
	"""
	STDP rule involving both pre- and post-synaptic spiking activity. As well
	as a homeostatic plasticity term depending on the firing rate. By default,
	pre-synaptic update is negative and the post-synaptic update is positive.
	"""

	def __init__(
		self,
		connection: AbstractConnection,
		nu: Optional[Union[float, Sequence[float]]] = None,
		reduction: Optional[callable] = None,
		weight_decay: float = 0.0,
		gamma: float = 0.005,
		constrain_nonnegative: bool = False,
		**kwargs,
	) -> None:
		"""
		Constructor for ``HomeostaticSTDP`` learning rule.
		:param connection: An ``AbstractConnection`` object whose weights the
			``PostPre`` learning rule will modify.
		:param nu: Single or pair of learning rates for pre- and post-synaptic events.
		:param reduction: Method for reducing parameter updates along the batch
			dimension.
		:param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
		:param gamma: Learning rate for homeostatic plasticity.
		:param constrain_nonnegative: Whether the weights are constrained to be nonnegative (>=0).
		"""
		super().__init__(
			connection=connection,
			nu=nu,
			reduction=reduction,
			weight_decay=weight_decay,
			**kwargs,
		)

		assert (self.source.homeostatic_traces and self.target.homeostatic_traces), \
			"Both pre- and post-synaptic nodes must record homeostatic spike traces. Use HomestaticLIFNodes."

		if isinstance(connection, (Connection, LocalConnection)):
			self.update = self._connection_update
		else:
			raise NotImplementedError("This learning rule is not supported for this Connection type.")
		
		self.gamma = gamma
		self.constrain_nonnegative = constrain_nonnegative

	def _connection_update(self, **kwargs) -> None:
		"""
		Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
		"""
		batch_size = self.source.batch_size

		# homeostatic update: delta w_ij = -gamma r_post w_ij
		self.connection.w -= self.reduction(self.gamma * self.target.r.view(batch_size, -1).unsqueeze(1) * self.connection.w, dim=0)

		# Pre-synaptic update.
		if self.nu[0]:
			source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
			target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
			self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
			del source_s, target_x

		# Post-synaptic update.
		if self.nu[1]:
			target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
			source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
			self.connection.w += self.reduction(torch.bmm(source_x, target_s), dim=0)
			del source_x, target_s

		super().update()

		if self.constrain_nonnegative:
			self.connection.w.clamp_(0)