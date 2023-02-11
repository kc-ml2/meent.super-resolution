import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_piecewise_constant, put_permittivity_in_ucell, read_material_table, \
    to_conv_mat
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot_zx


class RCWANumpy(_BaseRCWA):
    def __init__(self,
                 n_I=1.,
                 n_II=1.,
                 theta=0,
                 phi=0,
                 psi=0,
                 period=(100, 100),
                 wavelength=900,
                 ucell=None,
                 thickness=None,
                 perturbation=1E-10,
                 mode=0,
                 grating_type=0,
                 pol=0,
                 fourier_order=40,
                 ucell_materials=None,
                 algo='TMM',
                 device='cpu',
                 type_complex=np.complex128,
                 fft_type='default',
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi, pol=pol,
                         fourier_order=fourier_order, period=period, wavelength=wavelength,
                         ucell=ucell, ucell_materials=ucell_materials,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex,)

        self.mode = mode
        self.device = 'cpu'
        self.type_complex = type_complex
        self.fft_type = fft_type

        self.mat_table = read_material_table(type_complex=self.type_complex)
        self.layer_info_list = []

    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        self.kx_vector = self.get_kx_vector(wavelength)

        if self.grating_type == 0:
            de_ri, de_ti = self.solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti = self.solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti = self.solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri.real, de_ti.real

    def run_ucell(self):
        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        if self.fft_type == 'default':
            E_conv_all = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
            o_E_conv_all = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)
        elif self.fft_type == 'piecewise':
            E_conv_all = to_conv_mat_piecewise_constant(ucell, self.fourier_order, type_complex=self.type_complex)
            o_E_conv_all = to_conv_mat_piecewise_constant(1 / ucell, self.fourier_order, type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti = self.solve(self.wavelength, E_conv_all, o_E_conv_all)

        return de_ri, de_ti

    def calculate_field(self, resolution=None, plot=True):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d(self.wavelength, self.n_I, self.theta, self.fourier_order, self.T1,
                                       self.layer_info_list, self.period, self.pol, resolution=resolution,
                                       type_complex=self.type_complex)
        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d_conical(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order,
                                               self.T1,
                                               self.layer_info_list, self.period, resolution=resolution,
                                               type_complex=self.type_complex)

        else:
            resolution = [100, 100, 100] if not resolution else resolution
            t0 = time.time()
            field_cell = field_dist_2d(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
                                       self.layer_info_list, self.period, resolution=resolution,
                                       type_complex=self.type_complex)
            print(time.time() - t0)
        if plot:
            field_plot_zx(field_cell, self.pol)

        return field_cell
