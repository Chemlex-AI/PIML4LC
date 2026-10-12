from __future__ import annotations

import gc
import tempfile
import warnings
import os
import shutil

import numpy as np

from typing import Union
from numpy import ndarray
from scipy.interpolate import CubicSpline

from cadet import Cadet


__all__ = ['CadetSolver', 'reset_client']

def reset_client():
    gc.collect()

_cadet_path = os.environ.get('CADET_CLI_PATH') or shutil.which('cadet-cli')
if not _cadet_path:
    raise FileNotFoundError('Install CADET and put cadet-cli on PATH, or set CADET_CLI_PATH.')
Cadet.cadet_path = _cadet_path


class CadetSolve:
    def __init__(self, n_comp: int):
        self.n_comp = n_comp

    def __call__(
            self, *,
            end_con: ndarray, end_time: ndarray,
            sa: ndarray, sd: ndarray, sq: ndarray,
            ka: ndarray, kd: ndarray, qmax: ndarray,
            col_disp: ndarray, fil_diff: ndarray, par_diff: ndarray,
            n_time: int = 180,
            clip: bool, bul_out_only: bool,
            **kwargs
        ) -> Union[dict[str, ndarray], ndarray, None]:
        result = self.solve(
            end_con=end_con, end_time=end_time,
            sa=sa, sd=sd, sq=sq,
            ka=ka, kd=kd, qmax=qmax,
            col_disp=col_disp, fil_diff=fil_diff, par_diff=par_diff,
            n_time=n_time,
            **kwargs
        )

        if result:
            if clip:
                result = {
                    k: np.clip(v, a_min=0.0, a_max=None)
                    for k, v in result.items()
                }
            return result['bul'][:, -1] if bul_out_only else result
        return None

    
    def solve(
            self, *,
            end_con: ndarray, end_time: ndarray,
            sa: ndarray, sd: ndarray, sq: ndarray,
            ka: ndarray, kd: ndarray, qmax: ndarray,
            col_disp: ndarray, fil_diff: ndarray, par_diff: ndarray,
            n_time: int = 180, **kwargs
        ) -> Union[dict[str, ndarray], None]:
        raise NotImplementedError()


    def cadet_kwargs(self, n_comp: int) -> dict[str, dict]:
        cadet_input_model = {
            'nunits': 3,

            'unit_000': {
                'unit_type': 'INLET',
                'ncomp': n_comp,
                'inlet_type': 'PIECEWISE_CUBIC_POLY'
            },

            'unit_001': {
                'unit_type': 'GENERAL_RATE_MODEL',
                'ncomp': n_comp,
                ## Geometry
                'col_length': 0.03,                    # column length m
                'cross_section_area': 0.0015**2*np.pi, # m^2
                'col_porosity': 0.5,                   # column porosity
                'par_porosity': 0.4,                   # bead porosity
                'par_radius': 1.35e-6,                 # bead radius m
                'par_coreradius': 0.95e-6,
                ## Transport
                'col_dispersion_multiplex': 1,         # m^2 / s (interstitial volume)
                'par_surfdiffusion': [0.0] * n_comp,   # m^2 / s (solid phase)

                'adsorption_model': 'EXT_MULTI_COMPONENT_LANGMUIR',
                'adsorption': {
                    'is_kinetic': True,                 # Kinetic binding
                },

                
                'discretization': {
                    ## Grid cells
                    'ncol': 101, 'npar': 6,
                    ## Bound states
                    'nbound': [1] * n_comp,
                    ## Other options
                    'par_disc_type': 'EQUIDISTANT_PAR',
                    'use_analytic_jacobian': 1,
                    'reconstruction': 'WENO',
                    'gs_type': 1,
                    'max_krylov': 0,
                    'max_restarts': 10,
                    'schur_safety': 1.0e-8,

                    'weno': {
                        'boundary_model': 0,
                        'weno_eps': 1e-10,
                        'weno_order': 3
                    }
                }
            },

            'unit_002': {
                'unit_type': 'OUTLET',
                'ncomp': n_comp
            },

            'connections': {
                'nswitches': 1,
                'switch_000': {
                    'section': 0,
                    'connections': [
                        0, 1, -1, -1, 1e-8,  # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1 
                        1, 2, -1, -1, 1e-8  # [unit_001, unit_002, all components, all components, Q/ m^3*s^-1 
                    ]
                }
            },

            'solver': {
                'gs_type': 1,
                'max_krylov': 0,
                'max_restarts': 10,
                'schur_safety': 1e-8,
            }
        }

        cadet_solver = {
            # Number of cores for parallel simulation
            'nthreads': 0, 
            # Tolerances for the time integrator
            'time_integrator': {
                'abstol': 1e-6,
                'algtol': 1e-10,
                'reltol': 1e-6,
                'init_step_size': 1e-6,
                'max_steps': 1000000,
            }
        }


        return cadet_input_model, cadet_solver

    def _init_cadet(self):
        model = Cadet()
        cadet_input, cadet_solver = self.cadet_kwargs(n_comp=1)
        self._set_nested_attr(model.root.input.model, cadet_input)
        self._set_nested_attr(model.root.input.solver, cadet_solver)
        self._set_return(model.root.input['return'])
        return model

    def _cubic_spline(self, ts: ndarray, ys: ndarray) -> ndarray:
        # return: shape == (4(cube, quad, line, const), n_split)
        return CubicSpline(ts, ys).c

    def _set_return(self, cadet_return):
        cadet_return.split_components_data = 0
        cadet_return.split_ports_data = 0
        # cadet_return.unit_000.write_solution_bulk = 1
        cadet_return.unit_000.write_solution_inlet = 1
        cadet_return.unit_000.write_solution_outlet = 1

        # Copy settings to the other unit operations
        cadet_return.unit_001 = cadet_return.unit_000
        cadet_return.unit_002 = cadet_return.unit_000

        cadet_return.unit_001.write_solution_bulk = 1
        cadet_return.unit_001.write_solution_particle = 1
        cadet_return.unit_001.write_solution_solid = 1

    def _set_nested_attr(self, obj, att: dict):
        for key, value in att.items():
            if isinstance(value, dict):
                self._set_nested_attr(getattr(obj, key), value)
            else:
                setattr(obj, key, value)

    def _set_const(self, obj, p: ndarray, n_time: int):
        assert p.ndim == 1
        setattr(obj, 'extfun_type', 'PIECEWISE_CUBIC_POLY')
        for attr in ['cube_coeff', 'quad_coeff', 'lin_coeff']:
            setattr(obj, attr, np.array([0]))
        setattr(obj, 'const_coeff', p)
        setattr(obj, 'section_times', np.array([0, n_time]))
        setattr(obj, 'velocity', 1e-3 if self.n_comp == 1 else 1e3)

    def _set_cubic(
            self,
            obj,
            cubic_poly: ndarray, ts: ndarray,
            end_p: ndarray, n_time: int
        ):
        # cubic_poly.shape == (4(cube, quad, lin, const), n_split)
        assert cubic_poly.shape[0] == 4
        setattr(obj, 'extfun_type', 'PIECEWISE_CUBIC_POLY')
        attrs = ['cube_coeff', 'quad_coeff', 'lin_coeff']
        for poly, attr in zip(cubic_poly[:3], attrs):
            setattr(
                obj, attr,
                np.concatenate([poly, np.array([0.0])])
            )
        setattr(
            obj, 'const_coeff',
            np.concatenate([cubic_poly[3], end_p.reshape(1)])
        )
        setattr(obj, 'section_times', ts.tolist() + [n_time])

        setattr(obj, 'velocity', 1e-3 if self.n_comp == 1 else 1e3)



class CadetGradEluSolver(CadetSolve):
    def __init__(self):
        super().__init__(n_comp=1)

    def solve(
            self, *, 
            end_con: ndarray, end_time: ndarray,
            sa: ndarray, sd: ndarray, sq: ndarray,
            ka: ndarray, kd: ndarray, qmax: ndarray,
            col_disp: ndarray, fil_diff: ndarray, par_diff: ndarray,
            n_time: int = 180,
            **kwargs
        ) -> Union[dict[str, ndarray], None]:

        assert sa.shape == sd.shape == sq.shape == \
            ka.shape == kd.shape == qmax.shape == \
            col_disp.shape == fil_diff.shape == par_diff.shape
        assert len(sa) == 1

        #### Calculate cubic spline to approximate exp gradient
        # act_xxx.shape == (end_time + 1, )
        # end_xxx.shape == (1, ) invariant along time (n_time - end_time) else None

        # gradient active parameters, t = 0, ..., end_time
        ext_lin_eoeff = (end_con - 0.05) / end_time

        ext_prop = (0.05 + ext_lin_eoeff * np.linspace(0, end_time, end_time + 1))
        act_adsorp = np.exp(-sa * ext_prop)
        act_desorp = np.exp(-sd * ext_prop)
        act_qm     = np.exp(-sq * ext_prop)
        # gradient end parameters, t = end_time, ..., n_time
        if end_time < n_time:
            end_adsorp = np.exp(-sa * end_con)
            end_desorp = np.exp(-sd * end_con)
            end_qm     = np.exp(-sq * end_con)
        else:
            end_adsorp = None
            end_desorp = None
            end_qm = None
        # act_xxx_poly.shape == (4(cube, quad, lin, const), n_split) 
        ts = np.linspace(0, end_time, end_time // 10 + 1, dtype=int)
        act_adsorp_poly = self._cubic_spline(ts, act_adsorp[ts])
        act_desorp_poly = self._cubic_spline(ts, act_desorp[ts])
        act_qm_poly     = self._cubic_spline(ts, act_qm[ts])
        
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            model = self._init_cadet()

            model.root.input.model.unit_001.col_dispersion = col_disp
            model.root.input.model.unit_001.film_diffusion = fil_diff
            model.root.input.model.unit_001.par_diffusion  = par_diff

            self._set_cubic(
                model.root.input.model.external.source_000,
                act_adsorp_poly, ts,
                end_adsorp, n_time
            )
            self._set_cubic(
                model.root.input.model.external.source_001,
                act_desorp_poly, ts,
                end_desorp, n_time
            )
            self._set_cubic(
                model.root.input.model.external.source_002,
                act_qm_poly, ts,
                end_qm, n_time
            )

            model.root.input.model.unit_001.adsorption.extfun = [0, 1, 2]
            for para_name, para_value in zip(['ka', 'kd', 'qmax'], [ka, kd, qmax]):
                setattr(
                    model.root.input.model.unit_001.adsorption,
                    f'ext_mcl_{para_name}_t',
                    para_value
                )
                for order in ['', '_tt', '_ttt']:
                    setattr(
                        model.root.input.model.unit_001.adsorption,
                        'ext_mcl_' + para_name + order,
                        [0]
                    )
        
            model.root.input.model.unit_001.init_c = [0.0]
            model.root.input.model.unit_001.init_q = [0.0]

            model.root.input.solver.sections.nsec = 2
            model.root.input.solver.sections.section_times = [0.0, 0.4, n_time]
            model.root.input.solver.sections.section_continuity = [0,]
            model.root.input.model.unit_000.sec_000.const_coeff = np.array([1.0])
            model.root.input.model.unit_000.sec_001.const_coeff = [0]

            # Solution times
            model.root.input.solver.user_solution_times = np.linspace(0, n_time, n_time+1)

            model.filename = temp_file.name
            model.save()
            data = model.run()

            if data.returncode == 0:
                model.load()
                
                # t = model.root.output.solution.solution_times
                bul = model.root.output.solution.unit_001.solution_bulk[1:]
                par = model.root.output.solution.unit_001.solution_particle[1:]
                sol = model.root.output.solution.unit_001.solution_solid[1:]

                return {'bul': bul, 'par': par, 'sol': sol}
            else:
                warnings.warn(f'Cadet solve failed: {data.stdout}')




class CadetIsocEluSolver(CadetSolve):
    def __init__(self):
        super().__init__(n_comp=1)

    def solve(
            self, *,
            end_p: ndarray,
            sa: ndarray, sd: ndarray, sq: ndarray,
            ka: ndarray, kd: ndarray, qmax: ndarray,
            col_disp: ndarray, fil_diff: ndarray, par_diff: ndarray, 
            n_time: int = 180,
            **kwargs
        ) -> Union[dict[str, ndarray], None]:

        assert sa.shape == sd.shape == sq.shape == \
            ka.shape == kd.shape == qmax.shape == \
            col_disp.shape == fil_diff.shape == par_diff.shape
        assert len(sa) == 1

        adsorp = np.exp(-sa * end_p)
        desorp = np.exp(-sd * end_p)
        qm     = np.exp(-sq * end_p)
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            model = self._init_cadet()

            model.root.input.model.unit_001.col_dispersion = col_disp
            model.root.input.model.unit_001.film_diffusion = fil_diff
            model.root.input.model.unit_001.par_diffusion  = par_diff

            self._set_const(
                model.root.input.model.external.source_000,
                adsorp, n_time
            )
            self._set_const(
                model.root.input.model.external.source_001,
                desorp, n_time
            )
            self._set_const(
                model.root.input.model.external.source_002,
                qm, n_time
            )

            model.root.input.model.unit_001.adsorption.extfun = [0, 1, 2]
            for para_name, para_value in zip(['ka', 'kd', 'qmax'], [ka, kd, qmax]):
                setattr(
                    model.root.input.model.unit_001.adsorption,
                    f'ext_mcl_{para_name}_t',
                    para_value
                )
                for order in ['', '_tt', '_ttt']:
                    setattr(
                        model.root.input.model.unit_001.adsorption,
                        'ext_mcl_' + para_name + order,
                        [0]
                    )
        
            model.root.input.model.unit_001.init_c = [0.0] 
            model.root.input.model.unit_001.init_q = [0.0]

            model.root.input.solver.sections.nsec = 2
            model.root.input.solver.sections.section_times = [0.0, 0.4, n_time]
            model.root.input.solver.sections.section_continuity = [0,]
            model.root.input.model.unit_000.sec_000.const_coeff = np.array([1.0])
            model.root.input.model.unit_000.sec_001.const_coeff = [0]

            # Solution times
            model.root.input.solver.user_solution_times = np.linspace(0, n_time, n_time+1)

            model.filename = temp_file.name
            model.save()
            data = model.run()

            if data.returncode == 0:
                model.load()
                
                # t = model.root.output.solution.solution_times
                bul = model.root.output.solution.unit_001.solution_bulk[1:]
                par = model.root.output.solution.unit_001.solution_particle[1:]
                sol = model.root.output.solution.unit_001.solution_solid[1:]
                return {'bul': bul, 'par': par, 'sol': sol}
            else:
                warnings.warn(f'Cadet solve failed: {data.stdout}')





