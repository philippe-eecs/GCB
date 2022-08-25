import wandb
import os
os.environ["WANDB_API_KEY"] = '' #PLACE API KEY HERE
os.environ["MUJOCO_GL"] = 'osmesa'
from rlkit.core import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#We will have to replace logger...


class LoggingUtil():

	def __init__(self, details):

		if details['use_wandb']:
			try:
				#experiment_name = details['experiment_name']
				pass
			except:
				experiment_name = 'dry_run'
				os.environ["WANDB_MODE"] = 'offline'

			if details['offline_wandb']:
				os.environ["WANDB_MODE"] = 'offline'

			try:
				project_name = details['project_name']
			except:
				project_name = 'goalbisim' #Filler probably

			try:
				os.environ["WANDB_API_KEY"] = details['WANDB_API_KEY']
			except:
				os.environ["WANDB_API_KEY"] = ''

			wandb.login()

			try:
				wandb.init(project = project_name, group = details['group'], entity = details['entity'])
			except:
				wandb.init(project = project_name, group = details['group'], entity = '') #PLACE ENTIY HERE

			self.use_wandb = True

		else:
			self.use_wandb = False

		self.current_nonloaded_stats = {}
		self.current_images = [] #List of all 2D_Array images 


	def define_metric(self, metric):
		if self.use_wandb:
			wandb.define_metric(metric)


	def log_model(self, model):
		raise NotImplementedError

	def log(self, stats):
		'''
		Places Output 1D Stats into Local Queue to be Saved Eventually
		'''
		for key, item in stats.items():
			#assert not isinstance(item, matplotlib.figure.Figure()), "Please use record_figure to store matplotlib figures, not log"
			assert not isinstance(item, list), "Please use log_list to store lists, not log"
				
		try:
			logger.record_dict(stats) #Don't need to worry calling each time because each piece of data saved seperately anyway...
		except:
			pass
			
		self.current_nonloaded_stats.update(stats)

	def record(self, stats=None, step = None, with_prefix = True, with_timestamp = False):

		'''
		Saves current 1D stats on drive and uploads to wandb, wipes stats
		'''

		if stats is not None:
			self.log(stats)

		for img in self.current_images:
			self.current_nonloaded_stats.update({img[0] : img[1]})

		self.current_images = []


		try:

			if self.use_wandb:
				if step is not None:
					wandb.log(self.current_nonloaded_stats, step = step)
				else:
					wandb.log(self.current_nonloaded_stats)

				logger.dump_tabular(with_prefix=with_prefix, with_timestamp=with_timestamp)

			self.current_nonloaded_stats = {}

		except:
			pass #Wandb just needs a second...


	def log_figure(self, figure, save_name, figure_save_kwargs = None, wandb_save_loc = None, local_save_loc = None):
		'''
		Logs Matplotlib figure for saving as image
		'''

		#assert isinstance(figure, matplotlib.figure.Figure()), "Please feed in a matplotlib figure"

		if self.use_wandb:
			if wandb_save_loc is not None:
				self.current_images.append((wandb_save_loc, wandb.Image(plt)))

		if local_save_loc is not None:
			if local_save_loc[-1] != '/':
				local_save_loc += '/'

			if '/home/' in local_save_loc:

				figure.savefig(local_save_loc + save_name, **figure_save_kwargs)

			else:
				figure.savefig(logger.get_snapshot_dir() + local_save_loc + save_name, **figure_save_kwargs)

	def save_gif(self, step, file_name, frames, fps):
		if self.use_wandb:

			stats = {
			"train_step" : step,
			file_name : wandb.Video(frames, fps = fps, format = 'gif')
			}
			wandb.log(stats)


		#import pdb; pdb.set_trace()

		#import imageio
		#imageio.mimsave(logger.get_snapshot_dir() + '/' + file_name + str(step), frames, duration = fps)






