import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_raster_continuous, to_conv_mat_raster_discrete, to_conv_mat_vector
from .field_distribution import field_dist_1d_vectorized_ji, field_dist_1d_conical_vectorized_ji, field_dist_2d_vectorized_ji, field_plot, field_dist_1d_vanilla, \
    field_dist_1d_vectorized_kji, field_dist_1d_conical_vanilla, field_dist_1d_conical_vectorized_kji, \
    field_dist_2d_vectorized_kji, field_dist_2d_vanilla


class RCWANumpy(_BaseRCWA):
    def __init__(self,
                 n_I=1.,
                 n_II=1.,
                 theta=0.,
                 phi=0.,
                 period=(100., 100.),
                 wavelength=900.,
                 ucell=None,
                 ucell_info_list=None,
                 thickness=(0., ),
                 backend=0,
                 grating_type=0,
                 pol=0.,
                 fto=(2, 0),
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=np.complex128,
                 fft_type=0,
                 enhanced_dfs=True,
                 **kwargs,
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex, )

        self.ucell = ucell
        self.ucell_materials = ucell_materials
        self.ucell_info_list = ucell_info_list

        self.backend = backend
        self.fft_type = fft_type
        self.enhanced_dfs = enhanced_dfs

        self.layer_info_list = []

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):
        if isinstance(ucell, np.ndarray):
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            self._ucell = np.array(ucell, dtype=dtype)
        elif ucell is None:
            self._ucell = ucell
        else:
            raise ValueError

    def _solve(self, wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all):

        if self.grating_type == 0:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all)
        else:
            raise ValueError

        return de_ri, de_ti, layer_info_list, T1

    def solve(self, wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all):
        de_ri, de_ti, layer_info_list, T1 = self._solve(wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def conv_solve(self, **kwargs):
        # [setattr(self, k, v) for k, v in kwargs.items()]  # no need in npmeent

        if self.fft_type == 0:
            epx_conv_all, epy_conv_all, epz_i_conv_all = to_conv_mat_raster_discrete(self.ucell, self.fto[0], self.fto[1],
                                                                                     type_complex=self.type_complex, enhanced_dfs=self.enhanced_dfs)
        elif self.fft_type == 1:
            epx_conv_all, epy_conv_all, epz_i_conv_all = to_conv_mat_raster_continuous(self.ucell, self.fto[0], self.fto[1],
                                                                     type_complex=self.type_complex)
        elif self.fft_type == 2:
            epx_conv_all, epy_conv_all, epz_i_conv_all = to_conv_mat_vector(self.ucell_info_list, self.fto[0],
                                                          self.fto[1],
                                                          type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti, layer_info_list, T1 = self._solve(self.wavelength, epx_conv_all, epy_conv_all, epz_i_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def calculate_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):
        if self.grating_type == 0:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
                                                   self.T1, self.layer_info_list, self.period, self.pol,
                                                   res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector, self.T1, self.layer_info_list,
                                                         self.period, self.pol, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_vectorized_kji(self.wavelength, self.kx_vector, self.T1,
                                                          self.layer_info_list, self.period, self.pol,
                                                          res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            else:
                raise ValueError
        elif self.grating_type == 1:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                           self.phi, self.T1, self.layer_info_list, self.period,
                                                           res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                 self.phi, self.T1, self.layer_info_list, self.period,
                                                                 res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                  self.phi, self.T1, self.layer_info_list, self.period,
                                                                  res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            else:
                raise ValueError
        elif self.grating_type == 2:
            if field_algo == 0:
                field_cell = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                   self.fto[0], self.fto[1], self.T1, self.layer_info_list, self.period,
                                                   res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                         self.phi, self.fto[0], self.fto[1], self.T1, self.layer_info_list,
                                                         self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_2d_vectorized_kji(self.wavelength, self.n_I, self.theta,
                                                          self.phi, self.fto[0], self.fto[1], self.T1, self.layer_info_list,
                                                          self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                          type_complex=self.type_complex)
            else:
                raise ValueError
        else:
            raise ValueError
        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):
        de_ri, de_ti = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, field_algo=field_algo)
        return de_ri, de_ti, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)

    def calculate_field_all(self, res_x=20, res_y=20, res_z=20):
        t0 = time.time()
        field_cell0 = self.calculate_field(res_x=res_x, res_y=res_y, res_z=res_z, field_algo=0)
        print('no vector', time.time() - t0)
        t0 = time.time()
        field_cell1 = self.calculate_field(res_x=res_x, res_y=res_y, res_z=res_z, field_algo=1)
        print('ji vector', time.time() - t0)
        t0 = time.time()
        field_cell2 = self.calculate_field(res_x=res_x, res_y=res_y, res_z=res_z, field_algo=2)
        print('kji vector', time.time() - t0)

        print('gap(1-0): ', np.linalg.norm(field_cell1 - field_cell0))
        print('gap(2-1): ', np.linalg.norm(field_cell2 - field_cell1))
        print('gap(0-2): ', np.linalg.norm(field_cell0 - field_cell2))

        return field_cell0, field_cell1, field_cell2
