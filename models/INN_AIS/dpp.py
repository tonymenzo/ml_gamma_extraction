from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn
import torch.nn.functional as f
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
	importance_weighted_expectation, effective_sample_size_over_p, setup_quadratic_function

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class DPP(nn.Module, TargetDistribution):
	def __init__(self, dim, data = '/home/tonym/Code/data/Dbar_pipipi_SDP_data_5e6.npy', seed=0, n_test_set_samples=1000, use_gpu=True,
				 true_expectation_estimation_n_samples=int(1e7)):
		super(DPP, self).__init__()
		self.seed = seed
		self.dim = dim
		self.n_test_set_samples = n_test_set_samples
		self.data = data

		self.device = "cuda" if use_gpu else "cpu"
		self.to(self.device)

		# Generate 2D interpolation of the given data file

		# Load the data
		self.mp_thetap = np.load(self.data)
		# bin the data to approximate the density
		n_bins = 150
		H, xedges, yedges = np.histogram2d(self.mp_thetap[:,0], self.mp_thetap[:,1], n_bins, density = True, range = [[0., 1.], [0.,1.]])
		xedges_center = (xedges[:-1] + xedges[1:]) / 2
		yedges_center = (yedges[:-1] + yedges[1:]) / 2
		# Fill bins with zero-counts to small but non-zero number
		for i in range(n_bins):
			for j in range(n_bins):
				if H[i,j] == 0:
					H[i,j] = 1e-20
		# Interpolate over the full Dalitz plot (linear interpolation is the fastest)
		self.density = RegularGridInterpolator((xedges_center, yedges_center), H, bounds_error = False, fill_value = 1e-10, method = 'linear')
		
		self.expectation_function = quadratic_function
		self.register_buffer("true_expectation", MC_estimate_true_expectation(self,
															 self.expectation_function,
															 true_expectation_estimation_n_samples
																			  ))
		self.device = "cuda" if use_gpu else "cpu"
		self.to(self.device)

	def to(self, device):
		if device == "cuda":
			if torch.cuda.is_available():
				self.cuda()
		else:
			self.cpu()

	@property
	def test_set(self) -> torch.Tensor:
		return self.sample((self.n_test_set_samples, ))

	def log_prob(self, x: torch.Tensor):
		log_prob = torch.from_numpy(np.log(self.density(x)))
		log_prob = log_prob.float().requires_grad_(requires_grad=True)
		
		# Very low probability samples can cause issues (we turn off validate_args of the
		# distribution object which typically raises an expection related to this.
		# We manually decrease the distributions log prob to prevent them having an effect on
		# the loss/buffer.
		mask = torch.zeros_like(log_prob, dtype = torch.float)
		mask[log_prob < -1e4] = - torch.tensor(float("inf"), dtype=torch.float)
		log_prob = log_prob + mask
		return log_prob
	
	def get_subset(self, length):
		""" 
		Generate a random list of data points with length len from the data pool. This algorithm ensures that
		no duplicates are chosen, which becomes important when obtaining the CP conserving probability 
		distributions via the permutation method. This algorithm will need to be modified for cases in which 
		the len of the desired samples is ~ the size of the data pool. In these cases the algorithm reaches
		a point where it is often sampling repeat values and thus performs many unsuccessful trials. 

		data (string): path to dataset following the conventions described in the intialization of the class

		len (int): --- length of desired list of data points

		returns numpy array of m2 data 
		"""
		
		mp_thetap_subset = []
		#location = random.sample(range(self.pool_len), length)
		with open(data, 'r') as f:
			for i in range(length):
				if i == 0:
					location = random.randrange(self.pool_len)
					# Go to location in file and place marker
					f.seek(location)
					# Bound to partial line
					f.readline()
					# Split masses into list
					ls = f.readline().split()
					mp_thetap_subset.append([float(ls[0]), float(ls[1]), float(ls[2])])
				else:
					# Generate a trial mass value
					location = random.randrange(self.pool_len)
					# Go to location in file and place marker
					f.seek(location)
					# Bound to partial line
					f.readline()
					# Split masses into list
					ls = f.readline().split()
					try:
						mp_thetap_subset_i = [[float(ls[0]), float(ls[1]), float(ls[2])]]
					except:
						print('End of file reached at location ', location, '. Trying reducing the pool_len, retrying...')
						# Generate a trial mass value
						location = random.randrange(self.pool_len)
						# Go to location in file and place marker
						f.seek(location)
						# Bound to partial line
						f.readline()
						# Split masses into list
						ls = f.readline().split()
						mp_thetap_subset_i = [[float(ls[0]), float(ls[1]), float(ls[2])]]
					
					if not(any(i in mp_thetap_subset_i for i in mp_thetap_subset)):
						mp_thetap_subset.append(mp_thetap_subset_i[0])
					else:
						# You found a duplicate, loop until you find an entry which is unique
						while any(i in mp_thetap_subset_i for i in mp_thetap_subset):
							# Generate a trial mass value
							location = random.randrange(self.pool_len)
							# Go to location in file and place marker
							f.seek(location)
							# Bound to partial line
							f.readline()
							# Split masses into list
							ls = f.readline().split()
							try:
								mp_thetap_subset_i = [[float(ls[0]), float(ls[1]), float(ls[2])]]
							except:
								continue
						mp_thetap_subset.append(mp_thetap_subset_i[0])
			
			"""
			# This way works but is x10 slower than the above method
			m2 = random.sample(f.readlines(), length)
			# Transform into list of floats
			for i in range(length):
				ls = m2[i].split()
				m2[i] = np.array([float(ls[0]), float(ls[1]), float(ls[2])])
			"""
		return torch.from_numpy(np.array(mp_thetap_subset))

	def sample(self, length=1):
		"""
		# Generate uniformly sampled random variable
		rand = torch.rand(shape)
		print(rand)
		# Weight each point according to the 2D interpolating function
		weights = self.density(rand)
		weights = torch.from_numpy(weights).unsqueeze(-1)
		weights = weights.repeat(1,2)
		print(weights)
		rand = rand * weights
		"""
		# For now I will just need to give back a sample from the dataset
		#rand = np.random.uniform(size=(n_rand,2))
		#weights = mp_thetap_interp(rand)
		rng = np.random.default_rng()
		return torch.from_numpy(rng.choice(self.mp_thetap, length, replace = False))

	def evaluate_expectation(self, samples, log_w):
		expectation = importance_weighted_expectation(self.expectation_function,
														 samples, log_w)
		true_expectation = self.true_expectation.to(expectation.device)
		bias_normed = (expectation - true_expectation) / true_expectation
		return bias_normed

	def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
							log_q_fn: Optional[LogProbFunc] = None,
							batch_size: Optional[int] = None) -> Dict:
		bias_normed = self.evaluate_expectation(samples, log_w)
		bias_no_correction = self.evaluate_expectation(samples, torch.ones_like(log_w))
		if log_q_fn:
			log_q_test = log_q_fn(self.test_set)
			log_p_test = self.log_prob(self.test_set)
			test_mean_log_prob = torch.mean(log_q_test)
			kl_forward = torch.mean(log_p_test - log_q_test)
			ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
			summary_dict = {
				"test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
				"bias_normed": torch.abs(bias_normed).cpu().item(),
				"bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
				"ess_over_p": ess_over_p.detach().cpu().item(),
				"kl_forward": kl_forward.detach().cpu().item()
							}
		else:
			summary_dict = {"bias_normed": bias_normed.cpu().item(),
							"bias_no_correction": torch.abs(bias_no_correction).cpu().item()}
		return summary_dict
