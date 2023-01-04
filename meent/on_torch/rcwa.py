import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat, put_permittivity_in_ucell, read_material_table
from .field_distribution import field_dist_1d, field_dist_2d, field_plot_zx


class RCWALight(_BaseRCWA):
    def __init__(self, mode=0, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wavelength=np.linspace(900, 900, 1), pol=0, patterns=None, ucell=None, ucell_materials=None, thickness=None, algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wavelength, pol, patterns, ucell, ucell_materials,
                         thickness, algo)

        self.mode = mode
        self.spectrum_r, self.spectrum_t = None, None
        # self.init_spectrum_array()
        self.mat_table = read_material_table()

    def solve(self, wavelength, e_conv_all, o_e_conv_all):

        # TODO: !handle uniform layer

        if self.grating_type == 0:
            de_ri, de_ti = self.solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti = self.solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti = self.solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri.real, de_ti.real

    # def loop_wavelength_fill_factor_(self, wavelength_array=None):
    #
    #     if wavelength_array is not None:
    #         self.wavelength = wavelength_array
    #         self.init_spectrum_array()
    #
    #     for i, wavelength in enumerate(self.wavelength):
    #
    #         ucell = fill_factor_to_ucell(self.patterns, wavelength, self.grating_type, self.mat_table)
    #         e_conv_all = to_conv_mat(ucell, self.fourier_order)
    #         o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)
    #
    #         de_ri, de_ti = self.solve(wavelength, e_conv_all, o_e_conv_all)
    #         self.spectrum_r[i] = de_ri
    #         self.spectrum_t[i] = de_ti
    #
    #     return self.spectrum_r, self.spectrum_t
    #
    # def loop_wavelength_ucell_(self, wavelength_array=None):
    #
    #     if wavelength_array is not None:
    #         self.wavelength = wavelength_array
    #         self.init_spectrum_array()
    #
    #     for i, wavelength in enumerate(self.wavelength):
    #
    #         ucell = put_permittivity_in_ucell(self.ucell, wavelength, self.grating_type, self.mat_table)
    #         e_conv_all = to_conv_mat(ucell, self.fourier_order)
    #         o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)
    #
    #         de_ri, de_ti = self.solve(wavelength, e_conv_all, o_e_conv_all)
    #
    #         self.spectrum_r[i] = de_ri
    #         self.spectrum_t[i] = de_ti
    #
    #     return self.spectrum_r, self.spectrum_t

    def run_ucell(self):
        t0 = time.time()
        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength)
        t1 = time.time()
        e_conv_all = to_conv_mat(ucell, self.fourier_order)
        t2 = time.time()

        o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)
        t3 = time.time()

        de_ri, de_ti = self.solve(self.wavelength, e_conv_all, o_e_conv_all)
        t4 = time.time()
        print(t1 -t0, t2-t1,t3-t2,t4-t3)

        # ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength)
        #
        # e_conv_all = to_conv_mat(ucell, self.fourier_order)
        # o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)
        #
        # de_ri, de_ti = self.solve(self.wavelength, e_conv_all, o_e_conv_all)

        return de_ri, de_ti

    def calculate_field(self, resolution=None, plot=True):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d(self.wavelength, self.n_I, self.theta, self.fourier_order, self.T1,
                                self.layer_info_list, self.period, self.pol, resolution=resolution)
        else:
            resolution = [100, 100, 100] if not resolution else resolution
            field_cell = field_dist_2d(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
                                self.layer_info_list, self.period, resolution=resolution)

        if plot:
            field_plot_zx(field_cell, self.pol)

        return field_cell

