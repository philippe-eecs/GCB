import torch



def load_encoder(parameters):
	encoder_form = parameters['encoder_form']

	if encoder_form == 'RADEncoder':
		from goalbisim.encoders.RADencoder import RADPixelEncoder
		encoder = RADPixelEncoder((3 * parameters['temporal_frame_stack'], parameters['obs_height'], parameters['obs_width']), \
		 parameters['latent_size'], num_layers=parameters['encoder_num_layers'])

	else:
		raise NotImplementedError


	return encoder