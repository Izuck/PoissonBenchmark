"""Versioned mathematical definitions, independent of solver packages."""
import numpy as np

MOSCAP = {
    'schema_version': '1.0', 'problem_id': 'moscap_charge_free_v1',
    'description': 'Two dielectric layers with ideal gate and bottom contacts; no mobile/fixed charge.',
    'coordinates': {'domain': [[0.,1.],[0.,1.]], 'normalization_length_m': 1e-7,
                    'width_m': 1e-7, 'height_m': 1e-7, 'axis_order': 'solution[y_index,x_index]'},
    'interface_y_normalized': .9,
    'materials': {'silicon_relative_permittivity': 11.7, 'oxide_relative_permittivity': 3.9},
    'equation': '-div(epsilon_r grad(u))=0; u is in volts',
    'boundary_conditions': {'y=1':'gate_V', 'y=0':'bottom_V', 'x=0,x=1':'zero normal displacement flux'},
    'interface_conditions': ['continuous potential', 'continuous normal displacement flux'],
    'input_columns': ['gate_V','bottom_V'],
    'parameter_ranges': {'gate_V':[.2,2.], 'bottom_V':[-.2,.2]},
    'sampling': {'size':[512,512], 'location':'uniform cell centers','dtype':'float64',
                 'format':'uncompressed NumPy .npy', 'compression':False},
    'benchmark_threshold_relative_l2': .001,
    'precision_verification_absolute_limit_V': 1e-10,
    'not_included': ['mobile carriers','doping','work-function differences','interface charge',
                     'inversion/accumulation','source/drain','fringing','transport','quantum corrections'],
    'status': 'benchmark defaults, not final solver requirements',
}


def exact_moscap(x,y,coefficients):
    gate,bottom=coefficients
    h=MOSCAP['interface_y_normalized']
    si=MOSCAP['materials']['silicon_relative_permittivity']
    ox=MOSCAP['materials']['oxide_relative_permittivity']
    resistance=np.minimum(y,h)/si+np.maximum(y-h,0)/ox
    return bottom+(gate-bottom)*resistance/(h/si+(1-h)/ox)


def axes(size=512):
    return (np.arange(size,dtype=np.float64)+.5)/size


def metrics(prediction,reference):
    if prediction.shape!=reference.shape or not np.isfinite(prediction).all():
        raise ValueError('Prediction must have reference shape and only finite values')
    difference=prediction-reference
    norm=float(np.linalg.norm(reference))
    return {'relative_l2_eval':float(np.linalg.norm(difference)/norm) if norm else None,
            'rmse_V':float(np.sqrt(np.mean(difference*difference))),
            'max_abs_error_V':float(np.max(np.abs(difference)))}
