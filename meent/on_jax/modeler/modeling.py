from bisect import bisect_left

import jax.numpy as jnp
import numpy as np

from os import walk
from pathlib import Path


class ModelingJax:
    def __init__(self, *args, **kwargs):
        pass
        # self.ucell = None
        # self.ucell_vector = None
        # self.x_list = None
        # self.y_list = None
        # self.mat_table = None

    # def _tree_flatten(self):  # TODO
    #     children = (self.n_I, self.n_II, self.theta, self.phi, self.psi,
    #                 self.period, self.wavelength, self.ucell, self.ucell_info_list, self.thickness)
    #     aux_data = {
    #         'backend': self.backend,
    #         'grating_type': self.grating_type,
    #         'pol': self.pol,
    #         'fto': self.fourier_order,
    #         'ucell_materials': self.ucell_materials,
    #         'connecting_algo': self.algo,
    #         'perturbation': self.perturbation,
    #         'device': self.device,
    #         'type_complex': self.type_complex,
    #         'fourier_type': self.fft_type,
    #     }
    #
    #     return children, aux_data
    #
    # @classmethod
    # def _tree_unflatten(cls, aux_data, children):
    #     return cls(*children, **aux_data)

    @staticmethod
    def rectangle_no_approximation(cx, cy, lx, ly, base):

        a = [cy - ly / 2, cx - lx / 2]  # row, col
        b = [cy + ly / 2, cx + lx / 2]  # row, col

        res = [[a, b, base]]  # top_left, bottom_right

        return res

    def rectangle(self, cx, cy, lx, ly, n_index, angle=0, n_split_triangle=2, n_split_parallelogram=2,
                  angle_margin=1E-5):

        if type(lx) in (int, float):
            lx = jnp.array(lx).reshape(1)
        elif type(lx) is jnp.ndarray:
            lx = lx.reshape(1)

        if type(ly) in (int, float):
            ly = jnp.array(ly).reshape(1)
        elif type(ly) is jnp.ndarray:
            ly = ly.reshape(1)

        if type(angle) in (int, float):
            angle = jnp.array(angle).reshape(1)
        elif type(angle) is jnp.ndarray:
            angle = angle.reshape(1)

        if lx.dtype not in (jnp.complex64, jnp.complex128):
            if self.type_complex is jnp.complex128:
                lx = lx.astype(jnp.float64)
            else:
                lx = lx.astype(jnp.float32)

        if ly.dtype not in (jnp.complex64, jnp.complex128):
            if self.type_complex is jnp.complex128:
                ly = ly.astype(jnp.float64)
            else:
                ly = ly.astype(jnp.float32)
        # n_split_triangle, n_split_parallelogram = n_split_triangle + 2, n_split_parallelogram + 2

        # if angle is None:
        #     angle = jnp.array(0 * jnp.pi / 180)

        angle = angle % (2 * jnp.pi)

        # No rotation
        if 0 * jnp.pi / 2 - angle_margin <= abs(angle) % (2 * jnp.pi) <= 0 * jnp.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, lx, ly, n_index)
        elif 1 * jnp.pi / 2 - angle_margin <= abs(angle) % (2 * jnp.pi) <= 1 * jnp.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, ly, lx, n_index)
        elif 2 * jnp.pi / 2 - angle_margin <= abs(angle) % (2 * jnp.pi) <= 2 * jnp.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, lx, ly, n_index)
        elif 3 * jnp.pi / 2 - angle_margin <= abs(angle) % (2 * jnp.pi) <= 3 * jnp.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, ly, lx, n_index)
        else:
            pass

        # Yes rotation
        rotate = jnp.array([[jnp.cos(angle[0]), -jnp.sin(angle[0])], [jnp.sin(angle[0]), jnp.cos(angle[0])]],
                           dtype=self.type_complex)

        UR = rotate @ jnp.vstack([lx / 2, ly / 2])
        RD = rotate @ jnp.vstack([lx / 2, -ly / 2])
        DL = rotate @ jnp.vstack([-lx / 2, -ly / 2])
        LU = rotate @ jnp.vstack([-lx / 2, ly / 2])

        # UR += jnp.array([[cx], [cy]])
        # RD += jnp.array([[cx], [cy]])
        # DL += jnp.array([[cx], [cy]])
        # LU += jnp.array([[cx], [cy]])

        UR = UR.at[:].add([[cx], [cy]])
        RD = RD.at[:].add([[cx], [cy]])
        DL = DL.at[:].add([[cx], [cy]])
        LU = LU.at[:].add([[cx], [cy]])

        if 0 <= angle < jnp.pi / 2:
            angle_inside = (jnp.pi / 2) - angle

            # trail = L + U
            top1, top4 = UR, DL

            if LU[1].real > RD[1].real:
                top2, top3 = LU, RD
                length_top12, length_top24 = lx, ly
                top2_left = True
            else:
                top2, top3 = RD, LU
                length_top12, length_top24 = ly, lx
                top2_left = False

        elif jnp.pi / 2 <= angle < jnp.pi:

            angle_inside = jnp.pi - angle
            # trail = U + R
            top1, top4 = RD, LU

            if UR[1].real > DL[1].real:
                top2, top3 = UR, DL
                length_top12, length_top24 = ly, lx
                top2_left = True
            else:
                top2, top3 = DL, UR
                length_top12, length_top24 = lx, ly
                top2_left = False

        elif jnp.pi <= angle < jnp.pi / 2 * 3:
            angle_inside = (jnp.pi * 3 / 2) - angle

            # trail = R + D
            top1, top4 = DL, UR

            if RD[1].real > LU[1].real:
                top2, top3 = RD, LU
                length_top12, length_top24 = lx, ly
                top2_left = True
            else:
                top2, top3 = LU, RD
                length_top12, length_top24 = ly, lx
                top2_left = False

        elif jnp.pi / 2 * 3 <= angle < jnp.pi * 2:
            angle_inside = (jnp.pi * 2) - angle
            # trail = D + L
            top1, top4 = LU, RD

            if DL[1].real > UR[1].real:
                top2, top3 = DL, UR
                length_top12, length_top24 = ly, lx
                top2_left = True
            else:
                top2, top3 = UR, DL
                length_top12, length_top24 = lx, ly
                top2_left = False
        else:
            raise ValueError

        # point in region 1(top1~top2), 2(top2~top3) and 3(top3~top4)

        # xxx, yyy = [], []
        # xxx_cp, yyy_cp = [], []
        if top2_left:

            length = length_top12 / jnp.sin(angle_inside)
            top3_cp = [top3[0] - length, top3[1]]

            # for i in range(n_split_triangle + 1):
            #     x = top1[0] - (top1[0] - top2[0]) / n_split_triangle * i
            #     y = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * i
            #     xxx.append(x)
            #     yyy.append(y)
            #
            #     xxx_cp.append(x + length / n_split_triangle * i)
            #     yyy_cp.append(y)
            #
            # for i in range(n_split_parallelogram + 1):
            #
            #     x = top2[0] + (top3_cp[0] - top2[0]) / n_split_triangle * i
            #     y = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * i
            #     xxx.append(x)
            #     yyy.append(y)
            #
            #     xxx_cp.append(x + length)
            #     yyy_cp.append(y)
            #
            # for i in range(n_split_triangle + 1):
            #     x = top3_cp[0] + (top4[0] - top3_cp[0]) / n_split_triangle * i
            #     y = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * i
            #     xxx.append(x)
            #     yyy.append(y)
            #
            #     xxx_cp.append(x + length / n_split_triangle * (n_split_triangle - i))
            #     yyy_cp.append(y)

            # 1: Upper triangle
            xxx1 = top1[0] - (top1[0] - top2[0]) / n_split_triangle * jnp.arange(n_split_triangle + 1).reshape((-1, 1))
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * jnp.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            xxx_cp1 = xxx1 + length / n_split_triangle * jnp.arange(n_split_triangle + 1).reshape((-1, 1))
            yyy_cp1 = yyy1 * jnp.ones(n_split_triangle + 1).reshape((-1, 1))

            # 2: Mid parallelogram
            xxx2 = top2[0] + (top3_cp[0] - top2[0]) / n_split_triangle * jnp.arange(n_split_parallelogram + 1).reshape(
                (-1, 1))
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * jnp.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            xxx_cp2 = (xxx2 + length) * jnp.ones(n_split_parallelogram + 1).reshape((-1, 1))
            yyy_cp2 = yyy2 * jnp.ones(n_split_parallelogram + 1).reshape((-1, 1))

            # 3: Lower triangle
            xxx3 = top3_cp[0] + (top4[0] - top3_cp[0]) / n_split_triangle * jnp.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy3 = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * jnp.arange(
                n_split_triangle + 1).reshape(
                (-1, 1))

            xxx_cp3 = xxx3 + length / n_split_triangle * jnp.arange(n_split_triangle, -1, -1).reshape((-1, 1))
            yyy_cp3 = yyy3 * jnp.ones(n_split_triangle + 1).reshape((-1, 1))

            xxx = jnp.concatenate((xxx1, xxx2, xxx3))
            yyy = jnp.concatenate((yyy1, yyy2, yyy3))

            xxx_cp = jnp.concatenate((xxx_cp1, xxx_cp2, xxx_cp3))
            yyy_cp = jnp.concatenate((yyy_cp1, yyy_cp2, yyy_cp3))

            # # #####
            #
            # t00 = time.time()
            # obj_list1 = []
            #
            # for i in range(len(xxx)):
            #     if i == len(xxx) - 1:
            #         break
            #     x, y = xxx[i], yyy[i]
            #     x_cp, y_cp = xxx_cp[i], yyy_cp[i]
            #
            #     x_next, y_next = xxx[i + 1], yyy[i + 1]
            #     x_cp_next, y_cp_next = xxx_cp[i + 1], yyy_cp[i + 1]
            #
            #     x_mean = (x + x_next) / 2
            #     x_cp_mean = (x_cp + x_cp_next) / 2
            #     obj_list1.append([[y_cp_next, x_mean], [y, x_cp_mean], n_index])
            # t01 = time.time()
            #
            #
            # t0=time.time()
            # obj_list1 = []
            # x_mean_arr = (xxx + jnp.roll(xxx, -1)) / 2
            # x_cp_mean_arr = (xxx_cp + jnp.roll(xxx_cp, -1)) / 2
            # y_cp_next_arr = jnp.roll(yyy_cp, -1)
            #
            # for i in range(len(xxx)-1):
            #     obj_list1.append([[y_cp_next_arr[i], x_mean_arr[i]], [yyy[i], x_cp_mean_arr[i]], n_index])
            #
            # t1 =time.time()

            x_mean_arr = (xxx + jnp.roll(xxx, -1)) / 2
            x_cp_mean_arr = (xxx_cp + jnp.roll(xxx_cp, -1)) / 2
            y_cp_next_arr = jnp.roll(yyy_cp, -1)

            obj_list1 = [[[y_cp_next_arr[i], x_mean_arr[i]], [yyy[i], x_cp_mean_arr[i]], n_index] for i in
                         range(len(xxx) - 1)]

            # t2 =time.time()
            # print(t01-t00, t1-t0, t2-t1)

            # return obj_list1

        else:
            length = length_top12 / jnp.cos(angle_inside)
            top3_cp = [top3[0] + length, top3[1]]

            # 1: Top triangle
            xxx1 = top1[0] + (top2[0] - top1[0]) / n_split_triangle * jnp.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * jnp.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            xxx_cp1 = xxx1 - length / n_split_triangle * jnp.arange(n_split_triangle + 1).reshape((-1, 1))
            yyy_cp1 = yyy1 * jnp.ones(n_split_triangle + 1).reshape((-1, 1))

            # for i in range(n_split_triangle + 1):
            #     x = top1[0] + (top2[0] - top1[0]) / n_split_triangle * i
            #     y = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * i
            #     xxx.append(x)
            #     yyy.append(y)
            #
            #     xxx_cp.append(x - length / n_split_triangle * i)
            #     yyy_cp.append(y)

            # 2: Mid parallelogram
            xxx2 = top2[0] - (top2[0] - top3_cp[0]) / n_split_triangle * jnp.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * jnp.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            xxx_cp2 = xxx2 - length * jnp.ones(n_split_parallelogram + 1).reshape((-1, 1))
            yyy_cp2 = yyy2 * jnp.ones(n_split_parallelogram + 1).reshape((-1, 1))

            # for i in range(n_split_parallelogram + 1):
            #
            #     x = top2[0] - (top2[0] - top3_cp[0]) / n_split_triangle * i
            #     y = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * i
            #     xxx.append(x)
            #     yyy.append(y)
            #
            #     xxx_cp.append(x - length)
            #     yyy_cp.append(y)

            # 3: Lower triangle
            xxx3 = top3_cp[0] - (top3_cp[0] - top4[0]) / n_split_triangle * jnp.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy3 = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * jnp.arange(
                n_split_triangle + 1).reshape(
                (-1, 1))

            xxx_cp3 = xxx3 - length / n_split_triangle * jnp.arange(n_split_triangle, -1, -1).reshape((-1, 1))
            yyy_cp3 = yyy3 * jnp.ones(n_split_triangle + 1).reshape((-1, 1))

            xxx = jnp.concatenate((xxx1, xxx2, xxx3))
            yyy = jnp.concatenate((yyy1, yyy2, yyy3))

            xxx_cp = jnp.concatenate((xxx_cp1, xxx_cp2, xxx_cp3))
            yyy_cp = jnp.concatenate((yyy_cp1, yyy_cp2, yyy_cp3))

            # for i in range(n_split_triangle + 1):
            #     x = top3_cp[0] - (top3_cp[0] - top4[0]) / n_split_triangle * i
            #     y = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * i
            #     xxx.append(x)
            #     yyy.append(y)
            #
            #     xxx_cp.append(x - length / n_split_triangle * (n_split_triangle - i))
            #     yyy_cp.append(y)

            x_mean_arr = (xxx + jnp.roll(xxx, -1)) / 2
            x_cp_mean_arr = (xxx_cp + jnp.roll(xxx_cp, -1)) / 2
            y_cp_next_arr = jnp.roll(yyy_cp, -1)

            obj_list1 = [[[y_cp_next_arr[i], x_cp_mean_arr[i]], [yyy[i], x_mean_arr[i]], n_index] for i in
                         range(len(xxx) - 1)]

        return obj_list1

    def ellipse(self, cx, cy, lx, ly, n_index, angle=0, n_split_w=2, n_split_h=2, angle_margin=1E-5, debug=False):

        if type(lx) in (int, float):
            lx = jnp.array(lx).reshape(1)
        elif type(lx) is jnp.ndarray:
            lx = lx.reshape(1)

        if type(ly) in (int, float):
            ly = jnp.array(ly).reshape(1)
        elif type(ly) is jnp.ndarray:
            ly = ly.reshape(1)

        if type(angle) in (int, float):
            angle = jnp.array(angle).reshape(1)
        elif type(angle) is jnp.ndarray:
            angle = angle.reshape(1)

        if lx.dtype not in (jnp.complex64, jnp.complex128):
            if self.type_complex is jnp.complex128:
                lx = lx.astype(jnp.float64)
            else:
                lx = lx.astype(jnp.float32)

        if ly.dtype not in (jnp.complex64, jnp.complex128):
            if self.type_complex is jnp.complex128:
                ly = ly.astype(jnp.float64)
            else:
                ly = ly.astype(jnp.float32)
        angle = angle % (2 * jnp.pi)

        points_x_origin = lx / 2 * jnp.cos(jnp.linspace(jnp.pi / 2, 0, n_split_w))
        points_y_origin = ly / 2 * jnp.sin(jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_split_h))

        points_x_origin_contour = lx / 2 * jnp.cos(jnp.linspace(-jnp.pi, jnp.pi, n_split_w))[:-1]
        points_y_origin_contour = ly / 2 * jnp.sin(jnp.linspace(-jnp.pi, jnp.pi, n_split_h))[:-1]
        points_origin_contour = jnp.vstack([points_x_origin_contour, points_y_origin_contour])

        axis_x_origin = jnp.vstack([points_x_origin, jnp.ones(len(points_x_origin))])
        axis_y_origin = jnp.vstack([jnp.ones(len(points_y_origin)), points_y_origin])

        # rotate = jnp.ones((2, 2), dtype=points_x_origin.dtype)
        # rotate[0, 0] = jnp.cos(angle)
        # rotate[0, 1] = -jnp.sin(angle)
        # rotate[1, 0] = jnp.sin(angle)
        # rotate[1, 1] = jnp.cos(angle)
        #
        # rotate = rotate.at[(0, 0)].set(jnp.cos(angle))
        # rotate = rotate.at[(0, 1)].set(-jnp.sin(angle))
        # rotate = rotate.at[(1, 0)].set(jnp.sin(angle))
        # rotate = rotate.at[(1, 1)].set(jnp.cos(angle))

        rotate = jnp.array([[jnp.cos(angle[0]), -jnp.sin(angle[0])], [jnp.sin(angle[0]), jnp.cos(angle[0])]],
                           dtype=points_x_origin.dtype)

        axis_x_origin_rot = rotate @ axis_x_origin
        axis_y_origin_rot = rotate @ axis_y_origin

        axis_x_rot = axis_x_origin_rot[:, :, None]

        # axis_x_rot[0] += cx
        # axis_x_rot[1] += cy
        axis_x_rot = axis_x_rot.at[0, :, :].add(cx)
        axis_x_rot = axis_x_rot.at[1, :, :].add(cy)

        axis_y_rot = axis_y_origin_rot[:, :, None]

        # axis_y_rot[0] += cx
        # axis_y_rot[1] += cy
        axis_y_rot = axis_y_rot.at[0, :, :].add(cx)
        axis_y_rot = axis_y_rot.at[1, :, :].add(cy)

        points_origin_contour_rot = rotate @ points_origin_contour
        points_contour_rot = points_origin_contour_rot[:, :, None]

        # points_contour_rot[0] += cx
        # points_contour_rot[1] += cy
        points_contour_rot = points_contour_rot.at[0, :, :].add(cx)
        points_contour_rot = points_contour_rot.at[1, :, :].add(cy)

        y_highest_index = jnp.argmax(points_contour_rot.real, axis=1)[1, 0]

        points_contour_rot = jnp.roll(points_contour_rot, (points_contour_rot.shape[1] // 2 - y_highest_index).item(),
                                      axis=1)
        y_highest_index = jnp.argmax(points_contour_rot.real, axis=1)[1, 0]

        right = points_contour_rot[:, y_highest_index - 1]
        left = points_contour_rot[:, y_highest_index + 1]

        right_y = right[1].real
        left_y = left[1].real

        left_array = []
        right_array = []

        res = []

        if left_y > right_y:
            right_array.append(points_contour_rot[:, y_highest_index])
        elif left_y < right_y:
            left_array.append(points_contour_rot[:, y_highest_index])

        for i in range(points_contour_rot.shape[1] // 2):
            left_array.append(points_contour_rot[:, (y_highest_index + i + 1) % points_contour_rot.shape[1]])
            right_array.append(points_contour_rot[:, (y_highest_index - i - 1) % points_contour_rot.shape[1]])

        arr = jnp.zeros((2, len(right_array) + len(left_array), 1), dtype=points_contour_rot.dtype)

        if left_y > right_y:
            # arr[:, ::2] = jnp.stack(right_array, axis=1)
            # arr[:, 1::2] = jnp.stack(left_array, axis=1)

            arr = arr.at[:, ::2].set(jnp.stack(right_array, axis=1))
            arr = arr.at[:, 1::2].set(jnp.stack(left_array, axis=1))

        elif left_y < right_y:
            # arr[:, ::2] = jnp.stack(left_array, axis=1)
            # arr[:, 1::2] = jnp.stack(right_array, axis=1)

            arr = arr.at[:, ::2].set(jnp.stack(left_array, axis=1))
            arr = arr.at[:, 1::2].set(jnp.stack(right_array, axis=1))

        arr_roll = jnp.roll(arr, -1, 1)

        for i in range(arr.shape[1]):
            ax, ay = arr[:, i]
            bx, by = arr_roll[:, i]

            LL = [min(ay.real, by.real) + 0j, min(ax.real, bx.real) + 0j]
            UR = [max(ay.real, by.real) + 0j, max(ax.real, bx.real) + 0j]

            res.append([LL, UR, n_index])

        if debug:
            return res[:-1], (axis_x_rot, axis_y_rot, points_contour_rot)
        else:
            return res[:-1]

    def vector_per_layer_numeric(self, layer_info, x64=True):

        # TODO: activate and apply 'x64' option thru this function and connect to meent class.
        # TODO: make it clear: perturbation algorithm. For all backends.
        if x64:
            datatype = jnp.complex128
            perturbation = 0
            perturbation_unit = 1E-14
        else:
            datatype = jnp.complex64
            perturbation = 0
            perturbation_unit = 1E-6

        pmtvy_base, obj_list = layer_info

        # Griding
        row_list = []
        col_list = []

        # overlap check and apply perturbation
        for obj in obj_list:
            top_left, bottom_right, _ = obj

            # top_left[0]
            for _ in range(100):
                # index = bisect_left(row_list, top_left[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, top_left[0].real)
                if len(row_list) > index and top_left[0] == row_list[index]:
                    perturbation += perturbation_unit
                    if top_left[0] == 0:
                        top_left[0] = top_left[0] + perturbation

                    else:
                        top_left[0] = top_left[0] + (top_left[0] * perturbation)
                        # top_left = top_left.add[0].add(top_left[0] * perturbation)
                        # TODO: change; save how many perturbations were applied in a variable

                    row_list.insert(index, top_left[0])
                    break
                else:
                    row_list.insert(index, top_left[0])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(row_list, top_left[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, top_left[0].real)
                row_list.insert(index, top_left[0])

            # bottom_right[0]
            for _ in range(100):
                # index = bisect_left(row_list, bottom_right[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, bottom_right[0].real)
                if len(row_list) > index and bottom_right[0] == row_list[index]:
                    perturbation += perturbation_unit

                    # TODO: confirm assign makes right value
                    bottom_right[0] = bottom_right[0] - (bottom_right[0] * perturbation)
                    # bottom_right = bottom_right.at[0].add(-bottom_right[0] * perturbation)
                    row_list.insert(index, bottom_right[0])
                    break

                else:
                    row_list.insert(index, bottom_right[0])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(row_list, bottom_right[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, bottom_right[0].real)
                row_list.insert(index, bottom_right[0])

            # top_left[1]
            for _ in range(100):
                # index = bisect_left(col_list, top_left[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, top_left[1].real)
                if len(col_list) > index and top_left[1] == col_list[index]:
                    perturbation += perturbation_unit

                    if top_left[1] == 0:
                        # top_left = top_left.at[1].add(perturbation)
                        top_left[1] = top_left[1] + perturbation  # tODO
                    else:
                        top_left[1] = top_left[1] + (top_left[1] * perturbation)
                        # top_left = top_left.at[1].add(top_left[1] * perturbation)
                    col_list.insert(index, top_left[1])
                    break
                else:
                    col_list.insert(index, top_left[1])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(col_list, top_left[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, top_left[1].real)
                col_list.insert(index, top_left[1])

            # bottom_right[1]
            for _ in range(100):
                # index = bisect_left(col_list, bottom_right[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, bottom_right[1].real)
                if len(col_list) > index and bottom_right[1] == col_list[index]:
                    perturbation += perturbation_unit
                    # if bottom_right[1] == 0:
                    #     bottom_right[1] = bottom_right[1] + perturbation
                    # else:
                    #     # bottom_right[1] = bottom_right[1] + (bottom_right[1] * perturbation)
                    #     bottom_right[1] = bottom_right[1] - (bottom_right[1] * perturbation)

                    # bottom_right[1] = bottom_right[1] + (bottom_right[1] * perturbation)

                    # TODO: confirm assign makes right value
                    bottom_right[1] = bottom_right[1] - (bottom_right[1] * perturbation)
                    # bottom_right = bottom_right.at[1].add(-bottom_right[1] * perturbation)
                    col_list.insert(index, bottom_right[1])
                    break
                else:
                    col_list.insert(index, bottom_right[1])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(col_list, bottom_right[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, bottom_right[1].real)
                col_list.insert(index, bottom_right[1])

        if not row_list or row_list[-1] != self.period[1]:
            row_list.append(self.period[1].reshape(1).astype(datatype))
        if not col_list or col_list[-1] != self.period[0]:
            col_list.append(self.period[0].reshape(1).astype(datatype))

        if row_list and row_list[0] == 0:
            row_list = row_list[1:]
        if col_list and col_list[0] == 0:
            col_list = col_list[1:]

        ucell_layer = jnp.ones((len(row_list), len(col_list)), dtype=datatype) * pmtvy_base

        for obj in obj_list:
            top_left, bottom_right, pmty = obj

            if top_left[0] == 0:
                row_begin = 0
            else:
                row_begin = row_list.index(top_left[0]) + 1
            row_end = row_list.index(bottom_right[0]) + 1

            if top_left[1] == 0:
                col_begin = 0
            else:
                col_begin = col_list.index(top_left[1]) + 1
            col_end = col_list.index(bottom_right[1]) + 1

            ucell_layer = ucell_layer.at[row_begin:row_end, col_begin:col_end].set(pmty)

        x_list = jnp.concatenate(col_list).reshape((-1, 1))
        y_list = jnp.concatenate(row_list).reshape((-1, 1))

        return ucell_layer, x_list, y_list

    def draw(self, layer_info_list):
        ucell_info_list = []
        self.film_layer = jnp.zeros(len(layer_info_list))

        for i, layer_info in enumerate(layer_info_list):
            ucell_layer, x_list, y_list = self.vector_per_layer_numeric(layer_info)
            ucell_info_list.append([ucell_layer, x_list, y_list])
            if len(x_list) == len(y_list) == 1:
                # self.film_layer[i] = 1
                self.film_layer = self.film_layer.at[i].set(1)
        self.ucell_info_list = ucell_info_list
        return ucell_info_list

    def modeling_vector_instruction(self, instructions):

        # wavelength = rcwa_options['wavelength']

        # # Thickness update
        # t = rcwa_options['thickness']
        # for i in range(len(t)):
        #     if f'l{i + 1}_thickness' in fitting_parameter_name:
        #         t[i] = fitting_parameter_value[fitting_parameter_name[f'l{i + 1}_thickness']].reshape((1, 1))
        # mee.thickness = t

        # mat_table = read_material_table()

        # TODO: refractive index support string for nI and nII

        # Modeling
        layer_info_list = []
        for i, layer in enumerate(instructions):
            obj_list_per_layer = []
            base_refractive_index = layer[0]
            for j, vector_object in enumerate(layer[1]):
                func = getattr(self, vector_object[0])
                obj_list_per_layer += func(*vector_object[1:])

            layer_info_list.append([base_refractive_index, obj_list_per_layer])

        ucell_info_list = self.draw(layer_info_list)

        return ucell_info_list


def find_nk_index(material, mat_table, wl):
    if material[-6:] == '__real':
        material = material[:-6]
        n_only = True
    else:
        n_only = False

    mat_data = mat_table[material.upper()]
    n_index = jnp.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = jnp.interp(wl, mat_data[:, 0], mat_data[:, 2])
    nk = (n_index + 1j * k_index)

    return nk


def read_material_table(nk_path=None, type_complex=jnp.complex128):
    if type_complex == jnp.complex128:
        type_complex = jnp.float64
    elif type_complex == jnp.complex64:
        type_complex = jnp.float32
    else:
        raise ValueError

    mat_table = {}

    if nk_path is None:
        nk_path = str(Path(__file__).resolve().parent.parent.parent) + '/nk_data'

    full_path_list, name_list, _ = [], [], []
    for (dirpath, dirnames, filenames) in walk(nk_path):
        full_path_list.extend([f'{dirpath}/{filename}' for filename in filenames])
        name_list.extend(filenames)
    for path, name in zip(full_path_list, name_list):
        if name[-3:] == 'txt':
            data = np.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = type_complex(data)

        elif name[-3:] == 'mat':
            from scipy.io import loadmat
            data = loadmat(path)
            data = jnp.array([type_complex(data['WL']), type_complex(data['n']), type_complex(data['k'])])[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table
