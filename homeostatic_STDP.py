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

	def _connection_update(self, **kwargs) -> None:
		"""
		Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
		"""
		batch_size = self.source.batch_size

		# todo: fix this!
		# homeostatic update: delta w_ij = -gamma r_post w_ij
		if self.gamma:
			# self.connection.w -= self.reduction(self.gamma * self.target.r.view(batch_size, -1).unsqueeze(1) * self.connection.w, dim=0)
			self.connection.w -= self.reduction(self.gamma * self.target.r.view(batch_size, 1,self.target.n) * self.connection.w.view(batch_size, self.source.n, self.target.n), dim=0).reshape(*self.connection.w.shape)
		# need to broadcast first dim (over all pre)
		# w = 5 x 5 x 4
		# r = 5 x 4
		# normally -> 1 x 5 * 5 x 5 -> 5 x 5
		# r = 5 -> [1,5] -> [1,1,5]
		# w = 5 x 5

		super()._connection_update()