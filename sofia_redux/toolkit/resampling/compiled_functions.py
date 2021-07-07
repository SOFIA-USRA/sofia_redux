# Licensed under a 3-clause BSD style license - see LICENSE.rst

# import math
#
# import numpy as np
# import numba as nb
# from numba.pycc import CC
#
# cc = CC('compiled_functions')
# cc.target_cpu = 'host'
#
#
# @cc.export('scale_forward_vector',
#            'f8[:,:](f8[:,:],f8[:],f8[:])')
# def scale_forward_vector(coordinates, scale, offset):  # pragma: no cover
#
#     features, ndata = coordinates.shape
#     result = np.empty((features, ndata))
#     for k in range(features):
#         for i in range(ndata):
#             result[k, i] = (coordinates[k, i] - offset[k]) / scale[k]
#     return result
#
#
# @cc.export('scale_forward_scalar', 'f8[:](f8[:],f8[:],f8[:])')
# def scale_forward_scalar(coordinates, scale, offset):  # pragma: no cover
#     features = coordinates.size
#     result = np.empty(features)
#     for k in range(features):
#         result[k] = (coordinates[k] - offset[k]) / scale[k]
#     return result
#
#
# @cc.export('scale_reverse_vector', 'f8[:,:](f8[:,:],f8[:],f8[:])')
# def scale_reverse_vector(coordinates, scale, offset):  # pragma: no cover
#
#     features, ndata = coordinates.shape
#     result = np.empty((features, ndata))
#     for k in range(features):
#         for i in range(ndata):
#             result[k, i] = coordinates[k, i] * scale[k] + offset[k]
#     return result
#
#
# @cc.export('scale_reverse_scalar', 'f8[:](f8[:],f8[:],f8[:])')
# def scale_reverse_scalar(coordinates, scale, offset):  # pragma: no cover
#     features = coordinates.size
#     result = np.empty(features)
#     for k in range(features):
#         result[k] = coordinates[k] * scale[k] + offset[k]
#     return result
#
#
# @cc.export('matrix_power_product', 'f8[:](f8[:],i8[:,:])')
# def matrix_power_product(values, exponents):  # pragma: no cover
#     n_coeffs, n_dimensions = exponents.shape
#     pp = np.empty(n_coeffs)
#     for i in range(n_coeffs):
#         x = 1.0
#         for j in range(n_dimensions):
#             val = values[j]
#             exponent = exponents[i, j]
#             val_e = 1.0
#             for l in range(exponent):
#                 val_e *= val
#             x *= val_e
#         pp[i] = x
#     return pp
#
#
# @cc.export('tensor_power_product', 'f8[:,:](f8[:,:],i8[:,:])')
# def tensor_power_product(values, exponents):  # pragma: no cover
#     n_coeffs, n_dimensions = exponents.shape
#     n_data = values.shape[1]
#     pp = np.empty((n_coeffs, n_data))
#
#     for k in range(n_data):
#         for i in range(n_coeffs):
#             x = 1.0
#             for j in range(n_dimensions):
#                 val = values[j, k]
#                 exponent = exponents[i, j]
#                 val_e = 1.0
#                 for l in range(exponent):
#                     val_e *= val
#                 x *= val_e
#             pp[i, k] = x
#     return pp
#
#
# @cc.export('prune_1d', 'f8[:](f8[:],b1[:],i8)')
# def prune_1d(values, mask, n):  # pragma: no cover
#     result = np.empty(n)
#     found = 0
#     single_valued = values.size == 1
#     for i in range(mask.size):
#         if mask[i]:
#             if single_valued:
#                 result[found] = values[0]
#             else:
#                 result[found] = values[i]
#             found += 1
#     return result
#
#
# @cc.export('prune_2d', 'f8[:, :](f8[:, :],b1[:],i8)')
# def prune_2d(values, mask, n):  # pragma: no cover
#
#     nj = values.shape[0]
#     result = np.empty((nj, n))
#     found = 0
#     for i in range(mask.size):
#         if mask[i]:
#             for j in range(nj):
#                 result[j, found] = values[j, i]
#             found += 1
#     return result
#
#
# @cc.export('prune_equation_arrays',
#            'Tuple((f8[:], f8[:,:], f8[:], f8[:]))'
#            '(i8, b1[:], f8[:], f8[:,:], f8[:], f8[:])')
# def prune_equation_arrays(counts, equation_mask,
#                           equation_data, powers,
#                           equation_error,
#                           equation_dweights):  # pragma: no cover
#
#     ncoeffs, n = powers.shape
#     powerset = np.empty((ncoeffs, counts))
#     dataset = np.empty(counts)
#     errorset = np.empty(counts)
#     dweightset = np.empty(counts)
#
#     single_error = equation_error.size == 1
#     single_dweight = equation_dweights.size == 1
#
#     found = 0
#     for i in range(n):
#         if equation_mask[i]:
#             dataset[found] = equation_data[i]
#             for j in range(ncoeffs):
#                 powerset[j, found] = powers[j, i]
#             if single_error:
#                 errorset[found] = equation_error[0]
#             else:
#                 errorset[found] = equation_error[i]
#             if single_dweight:
#                 dweightset[found] = equation_dweights[0]
#             else:
#                 dweightset[found] = equation_dweights[i]
#             found += 1
#     return dataset, powerset, errorset, dweightset
#
#
# @cc.export('mask_counts2d', 'i8[:](b1[:,:])')
# def mask_counts2d(mask):  # pragma: no cover
#     n_sets, n = mask.shape
#     counts = np.empty(n_sets, dtype=nb.int64)
#     for i in range(n_sets):
#         count = 0
#         for j in range(n):
#             if mask[i, j]:
#                 count += 1
#         counts[i] = count
#     return counts
#
#
# # njit version is much faster
# @cc.export('weighted_mean', 'UniTuple(f8, 2)(f8[:],f8[:])')
# def weighted_mean(data, weights):  # pragma: no cover
#     n = data.size
#     dsum = 0.0
#     wsum = 0.0
#     for i in range(n):
#         dsum += data[i] * weights[i]
#         wsum += weights[i]
#
#     return dsum / wsum, 1.0 / wsum
#
#
# @cc.export('masked_mean', 'Tuple((f8, i8))(f8[:], b1[:])')
# def masked_mean(data, mask):  # pragma: no cover
#     nd = data.size
#     counts = 0
#     dsum = 0.0
#     for i in range(nd):
#         if mask[i]:
#             counts += 1
#             dsum += data[i]
#     if counts == 0:
#         return 0.0, 0
#     else:
#         return dsum / counts, counts
#
#
# @cc.export('masked_variance', 'Tuple((f8, i8))(f8[:], b1[:])')
# def masked_variance(data, mask):  # pragma: no cover
#     nd = data.size
#     counts = 0
#     dsum = 0.0
#     for i in range(nd):
#         if mask[i]:
#             counts += 1
#             dsum += data[i]
#     if counts < 2:
#         return 0.0, counts
#
#     dmean = dsum / counts
#     tmp = 0.0
#     for i in range(nd):
#         if mask[i]:
#             d = data[i] - dmean
#             d *= d
#             tmp += d
#     return tmp / (counts - 1), counts
#
#
# @cc.export('linear_system_solve',
#            'f8(f8[:,:],f8[:],f8[:],f8[:])')
# def linear_system_solve(powerset, dataset,
#                         weightset, visitor_power):  # pragma: no cover
#
#     ncoeffs, ndata = powerset.shape
#     alpha = np.empty((ncoeffs, ndata))
#     beta = np.empty(ncoeffs)
#     amat = np.empty((ncoeffs, ncoeffs))
#
#     for i in range(ncoeffs):
#         b = 0.0
#         for k in range(ndata):
#             w = weightset[k]
#             wa = w * powerset[i, k]
#             b += wa * w * dataset[k]
#             alpha[i, k] = wa
#         beta[i] = b
#
#     for i in range(ncoeffs):
#         for j in range(i, ncoeffs):
#             asum = 0.0
#             for k in range(ndata):
#                 asum += alpha[i, k] * alpha[j, k]
#             amat[i, j] = asum
#             if i != j:
#                 amat[j, i] = asum
#
#     coefficients = np.linalg.solve(amat, beta)
#     result = 0.0
#     for i in range(ncoeffs):
#         result += coefficients[i] * visitor_power[i]
#     return result
#
#
# @cc.export('cull_members', 'i8[:](f8[:, :], i8[:], i8[:], i8[:], b1)')
# def cull_members(offsets, members, multiplier,
#                  delta, return_indices):  # pragma: no cover
#
#     features = delta.size
#     for k in range(features):
#         if multiplier[k] != 0:
#             break
#         elif delta[k] != 0:
#             break
#     else:
#         if return_indices:
#             result = np.empty(members.size, dtype=nb.i8)
#             for i in range(members.size):
#                 result[i] = i
#             return result
#         else:
#             return members
#
#     nmembers = members.size
#     keep = np.empty(nmembers, dtype=nb.i8)
#     nfound = 0
#     for i in range(nmembers):
#         d = 0.0
#         member = members[i]
#         offset = offsets[:features, member]
#         for k in range(features):
#             doff = multiplier[k] * offset[k] + delta[k]
#             doff *= doff
#             d += doff
#             if d > 1:
#                 break
#         else:
#             if return_indices:
#                 keep[nfound] = i
#             else:
#                 keep[nfound] = member
#
#             nfound += 1
#
#     return keep[:nfound]
#
#
# @cc.export('calculate_equation_weights',
#            'UniTuple(f8[:], 2)(f8[:],f8[:],b1)')
# def calculate_equation_weights(errorset, dweightset,
#                                calculate_variance):  # pragma: no cover
#     # inverts it too
#     n = errorset.size
#     weights = np.empty(n)
#     if calculate_variance:
#         weights2 = np.empty(n)
#     else:
#         weights2 = weights
#
#     for i in range(n):
#         e = 1.0 / errorset[i]
#         weights[i] = e / dweightset[i]
#         if calculate_variance:
#             weights2[i] = weights[i] * e
#     return weights, weights2
#
#
# @cc.export('get_fit_range',
#            'Tuple((f8[:], i8[:]))(f8, f8[:,:], b1[:,:])')
# def get_fit_range(threshold, data, mask):  # pragma: no cover
#     n_sets, n = mask.shape
#     sigma = np.empty(n_sets)
#     counts = np.empty(n_sets, dtype=nb.int64)
#     for i in range(n_sets):
#         count = 0
#         for j in range(n):
#             if mask[i, j]:
#                 count += 1
#         counts[i] = count
#
#     if not threshold:
#         for i in range(n_sets):
#             sigma[i] = np.inf
#         return sigma, counts
#
#     for i in range(n_sets):
#         count = counts[i]
#         if count < 2:
#             sigma[i] = np.inf
#             continue
#
#         dmean = 0.0
#         for j in range(n):
#             if mask[i, j]:
#                 dmean += data[i, j]
#         dmean /= count
#
#         dsum = 0.0
#         for j in range(n):
#             if mask[i, j]:
#                 d = data[i, j] - dmean
#                 d *= d
#                 dsum += d
#         dsum /= count - 1
#         sigma[i] = math.sqrt(dsum) * threshold
#
#     return sigma, counts
#
#
# @cc.export('get_fit_range_with_error',
#            'Tuple((f8[:], i8[:]))(f8, f8[:,:], f8[:,:], b1[:,:])')
# def get_fit_range_with_error(threshold, data,
#                              error, mask):  # pragma: no cover
#     n_sets, n = mask.shape
#     sigma = np.empty(n_sets)
#     counts = np.empty(n_sets, dtype=nb.int64)
#     for i in range(n_sets):
#         count = 0
#         for j in range(n):
#             if mask[i, j]:
#                 count += 1
#         counts[i] = count
#
#     if not threshold:
#         for i in range(n_sets):
#             sigma[i] = np.inf
#         return sigma, counts
#
#     for i in range(n_sets):
#         count = counts[i]
#         if count < 2:
#             sigma[i] = np.inf
#             continue
#
#         dmean = 0.0
#         emean = 0.0
#         for j in range(n):
#             if mask[i, j]:
#                 dmean += data[i, j]
#                 emean += error[i, j]
#         dmean /= count
#         emean /= count
#
#         dsum = 0.0
#         for j in range(n):
#             if mask[i, j]:
#                 d = data[i, j] - dmean
#                 d *= d
#                 dsum += d
#         dsum /= count - 1
#         dsig = math.sqrt(dsum)
#         if dsig < emean:
#             dsig = emean
#
#         sigma[i] = dsig * threshold
#
#     return sigma, counts
#
#
# @cc.export('asymmetric_distance_weights', 'f8[:](f8[:,:], f8[:])')
# def asymmetric_distance_weights(offsets, alpha):  # pragma: no cover
#     dimensions, n = offsets.shape
#     weights = np.empty(n)
#     for i in range(n):
#         e = 0.0
#         for k in range(dimensions):
#             d = offsets[k, i] / alpha[k]
#             d *= d
#             e += d
#         weights[i] = 1.0 / math.exp(-e)
#     return weights
#
#
# @cc.export('symmetric_distance_weights', 'f8[:](f8[:], f8)')
# def symmetric_distance_weights(distance, alpha):  # pragma: no cover
#     """w = 1 / exp(-r^2 / alpha^2)"""
#     n = distance.size
#     weights = np.empty(n)
#     for i in range(n):
#         e = distance[i] / alpha
#         e *= e
#         weights[i] = 1.0 / math.exp(-e)
#
#     return weights
#
#
# # slower here than in njit
# @cc.export('offsets_from_center', 'f8[:,:](f8[:,:], f8[:])')
# def offsets_from_center(coordinates, center):  # pragma: no cover
#     features, npoints = coordinates.shape
#     offsets = np.empty((features, npoints))
#     for k in range(features):
#         for i in range(npoints):
#             offsets[k, i] = coordinates[k, i] - center[k]
#     return offsets
#
#
# # slower here that with njit
# @cc.export('line_max_order', 'i8(i8, f8, f8[:], b1)')
# def line_max_order(order, center, coordinates, required):  # pragma: no cover
#     left = 0
#     right = 0
#     left_found = False
#     right_found = False
#     unique_left = np.empty(order)
#     unique_right = np.empty(order)
#
#     for i in range(coordinates.size):
#         offset = coordinates[i] - center
#         if offset < 0:
#             if left_found:
#                 continue
#             elif left == 0:
#                 unique_left[0] = offset
#                 left = 1
#             else:
#                 for j in range(left):
#                     if unique_left[j] == offset:
#                         break
#                 else:
#                     unique_left[left] = offset
#                     left += 1
#             if left >= order:
#                 left_found = True
#
#         elif offset > 0:
#             if right_found:
#                 continue
#             elif right == 0:
#                 unique_right[0] = offset
#                 right = 1
#             else:
#                 for j in range(right):
#                     if unique_right[j] == offset:
#                         break
#                 else:
#                     unique_right[right] = offset
#                     right += 1
#             if right >= order:
#                 right_found = True
#
#         if left_found and right_found:
#             return order
#     else:
#         if required:
#             return -1
#         elif left < right:
#             return left
#         else:
#             return right
#
#
# @cc.export('symmetric_counts_maxorder', 'i8[:](i8[:], i8, i8, b1)')
# def symmetric_counts_maxorder(counts_array, order,
#                               features, required):  # pragma: no cover
#     npoints = counts_array.size
#     maxorder = np.empty(npoints, dtype=nb.i8)
#     minimum_points = (order + 1) ** features
#     for i in range(npoints):
#         if counts_array[i] >= minimum_points:
#             maxorder[i] = order
#         else:
#             limit = counts_array[i] ** (1.0 / features) - 1
#             if limit >= order:
#                 limit = order
#             elif required:
#                 limit = -1
#             elif limit < 0:
#                 limit = 0
#             else:
#                 limit = nb.i8(limit)
#
#             maxorder[i] = limit
#
#     return maxorder
#
#
# @cc.export('asymmetric_counts_maxorder', 'i8[:,:](i8[:], i8[:], i8)')
# def asymmetric_counts_maxorder(counts_array,
#                                order, features):  # pragma: no cover
#     # Just reject if it doesn't contain enough points
#     npoints = counts_array.size
#     maxorder = np.empty((features, npoints), dtype=nb.i8)
#     minimum_points = 1
#     for i in range(features):
#         minimum_points *= order[i] + 1
#
#     for i in range(npoints):
#         if counts_array[i] >= minimum_points:
#             for j in range(features):
#                 maxorder[j, i] = order[j]
#         else:
#             for j in range(features):
#                 maxorder[j, i] = -1
#
#     return maxorder
#
#
# @cc.export('mask_count', 'i8[:](b1[:,:])')  # njit version 2x speed
# def mask_count(mask):  # pragma: no cover
#     n_sets, n = mask.shape
#     counts = np.empty(n_sets, dtype=nb.int64)
#     for i in range(n_sets):
#         count = 0
#         for j in range(n):
#             if mask[i, j]:
#                 count += 1
#         counts[i] = count
#     return counts
#
#
# if __name__ == "__main__":
#     cc.compile()  # pragma: no cover
