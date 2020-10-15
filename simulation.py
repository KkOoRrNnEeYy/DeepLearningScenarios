import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import interpolation, continuation_parameter, interpolate_wb, augmented_frechet_matrix, analyse_point
from ODEnet import ODEnet
from additional_functions import *

def make_filename(params, cur_param):
    filename = f"MODEL~{cur_param}~"
    for k, i in params.items():
        filename += (str(k) + "=" + str(np.round(i, 5))+ ";")
    return filename

def make_fieldnames(params, n_sv):
    fieldnames = list(params.keys())
    fieldnames.append('det')
    fieldnames.extend([f'sv{i}' for i in range(n_sv)])
    return fieldnames

    
def make_info(fieldnames, model, det, sv):
    data = [*model.HP.values(), det, *sv]
    info = {}
    for i in range(len(fieldnames)):
        info[fieldnames[i]] = data[i]
    return info

def plot_model(model, parameter, z0, zs):
    z = np.array(model.Z).T
    a = np.array(model.A).T
    colors = ['red', 'green']
    t = np.linspace(0, model.HP['T'], len(z[0]))
    plt.title(f"Z  {parameter} = {model.HP[parameter]}")
    for i in range(len(z)):
        plt.plot(t, z[i], label=f"Z{i}", color=colors[i])
        plt.scatter([0,t[-1]], [z0[i], zs[i]], color=colors[i])
    plt.show()
    plt.title(f"A  {parameter} = {model.HP[parameter]}")
    for i in range(len(a)):
        plt.plot(t, a[i], label=f"A{i}")
    plt.show()
def plot_wb(model, parameter):
    t = np.linspace(0, model.HP['T'], len(model.Z))
    ws = [[] for i in range(model.HP['N']**2)]
    bs = [[] for i in range(model.HP['N'])]
    for l in model.wb:
        w = l[0]
        b = l[1]
        for i, wi in enumerate(np.nditer(w)):
            ws[i].append(wi)
        for bi in range(len(b)):
            bs[bi].append(b[bi])
    wb = [*ws, *bs]
    for i in range(len(wb)):
        if i < model.HP['N']**2:
            plt.title(f"w{i+1}~{parameter}={model.HP[parameter]}")
        else:
            title='b'
            plt.title(f"b{i-3}~{parameter}={model.HP[parameter]}")
        if i == 4:
            plt.plot(t, wb[i])
            plt.show()

def gen_model(hyper_parameters):
    nn = ODEnet(hyper_parameters, [hyper_parameters['N'] for i in range(int(HP['T']*10)+1)])
    return nn

def normal_net_training(model, epochs, z0, zs):
    p = epochs // 10
    for epoch in range(epochs):
        if epoch % p == 0 and epoch != 0:
#             print(f'{epoch}: {nn.Z[-1]}')
            model.plot_z(z0, zs, epoch)
        model.forward(z0)
        model.back_propagation(zs)
    model.forward(z0)
    
def boundary_net_training(model, precision, za0, z0, zs, parameter):
    epoch = 0
    while True:
        if epoch % 10 == 0 and epoch != 0:
            model.plot_z(z0, zs, epoch, parameter, det)
        newton_steps, det, sv = model.augmented_propagation(za0, z0, zs)
        za0 = np.concatenate((model.Z[len(model.Z)//2], model.A[len(model.A)//2]))
        if all(np.abs(model.Z[-1][i] - zs[i]) < precision for i in range(len(zs))):
            model.matrix_analyses(za0, z0, zs)
            break
        epoch += 1
    return newton_steps, det, sv

def biffurcation_analysis(model, parameter, step, until, precision=0.01):
    z0 = np.random.uniform(-1, 1, model.HP['N'])
    zs = np.random.uniform(-1, 1, model.HP['N'])
    za0 = np.random.uniform(-0.5, 0.5, model.HP['N']*2)

    zas = []
    ps = []
    
    fieldnames = make_fieldnames(model.HP, len(za0))
    file = make_filename(model.HP, parameter)
    create_csv(file, fieldnames)
    while True:
        newton_steps, det, sv = boundary_net_training(model, precision, za0, z0, zs, parameter)
        write_csv(file,  make_info(fieldnames, model, det, sv))
        za0 = continuation_parameter(ps, zas, model.HP[parameter], za0, step)
        model.update_parameter(parameter, step)
        model.reset_weights()
        if model.HP['E2'] < until or det > 1000:
            break
    plt.show()

def analyse_points(model, precision, parameter, p_values):
    z0 = np.random.uniform(-1, 1, model.HP['N'])
    zs = np.random.uniform(-1, 1, model.HP['N'])
    za0 = np.random.uniform(-0.5, 0.5, model.HP['N']*2)
    for i in range(len(p_values)):
        model.HP[parameter] = p_values[i]
        boundary_net_training(model, precision, za0, z0, zs, parameter)
    
HP = {'T': 10, 'LR': 0.0005, 'E1': 1, 'E2': 0, 'N': 4}
nn = gen_model(HP)
biffurcation_analysis(nn, 'E2', 0.001, -10)
