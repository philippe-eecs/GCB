import torch
import torch.nn as nn
import abc
from goalbisim.data.manipulation.transform import initialize_transform

class RLRepresentation(torch.nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, obs, detach = False):
		pass

	def encode(self, obs, detach = False):
		pass

	def update(self, replay_buffer, policy, kwargs, step, log = True):
		pass


def initialize_representation(obs_shape, action_shape, device, details, main = False):
	alg = details['representation_algorithm']

	if alg == 'Pixel':
		from goalbisim.representation.pixel import PixelRepresentation

		representation = PixelRepresentation(obs_shape, device, **details['pixel_kwargs'])

	elif alg == 'RAD':
		from goalbisim.representation.pixel import PixelRepresentation

		representation = PixelRepresentation(obs_shape, device, **details['rad_kwargs'])

	elif alg == 'CURL':
		from goalbisim.representation.curl import CURL

		representation = CURL(obs_shape, device, **details['curl_kwargs'])

	elif alg == 'GoalBiSim':
		from goalbisim.representation.goalbisim import GoalBisim
		
		#if main:
		representation = GoalBisim(obs_shape, device, action_shape = action_shape, **details['goalbisim_kwargs'])
		#else:
			#representation = GoalBisim(obs_shape, device, action_shape = action_shape, **details['goalbisim_kwargs'], using_phi = False)

	elif alg == 'Pairedencoder':
		from goalbisim.representation.pairedstategoal import PairedStateGoal

		representation = PairedStateGoal(obs_shape, device, action_shape = action_shape, **details['goalbisim_kwargs'])

	elif alg == "Vqvae":
		from goalbisim.representation.vqvae import VQVAE

		representation = VQVAE(obs_shape, device, **details['goalbisim_kwargs'])

	elif alg == "Ccvae":
		from goalbisim.representation.ccvae import CCVAE

		representation = CCVAE(obs_shape, device, **details['goalbisim_kwargs'])

	elif alg == "Vae":
		from goalbisim.representation.vae import VAE

		representation = VAE(obs_shape, device, **details['goalbisim_kwargs'])

	else:
		raise NotImplementedError

	return representation
