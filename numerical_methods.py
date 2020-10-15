import numpy as np
from numpy.linalg import det, inv, svd


def interpolation(Xs, Ys, x_next):
    a0 = Ys[0]
    if len(Ys) == 1: return a0
    a1 = (Ys[1] - Ys[0])/(Xs[1]-Xs[0])
    if len(Ys) == 2: return a0 + a1*(x_next-Xs[0])
    
    yx1x2 = (Ys[2]-Ys[1])/(Xs[2]-Xs[1])
    a2 = (yx1x2 - a1)/(Xs[2]-Xs[0])
    return a0 + a1*(x_next-Xs[0]) + a2*(x_next-Xs[0])*(x_next-Xs[1])

def continuation_parameter(Xs, Ys, x_cur, y_cur, step):
    x_next = x_cur + step
    if len(Ys) < 3:
        Ys.append(y_cur)
        Xs.append(x_cur)
    else:
        Ys[0], Ys[1], Ys[2] = Ys[1], Ys[2], y_cur
        Xs[0], Xs[1], Xs[2] = Xs[1], Xs[2], x_cur
    y_next = interpolation(Xs, Ys, x_next)
    return y_next

def interpolate_wb(ps, wbs, p_cur, wb, step):
    if len(wbs) < 3:
        wbs.append(wb)
        ps.append(p_cur)
    else:
        wbs[0], wbs[1], wbs[2] = wbs[1], wbs[2], wb.copy()
        ps[0], ps[1], ps[2] = ps[1], ps[2], p_cur

    for t in range(len(wb)):
        if len(wbs) == 1:
            cur_ws = [wbs[0][t][0]]
            cur_bs = [wbs[0][t][1]]
        if len(wbs) == 2:
            cur_ws = [wbs[0][t][0], wbs[1][t][0]]
            cur_bs = [wbs[0][t][1], wbs[1][t][1]]
        if len(wbs) == 3:
            cur_ws = [wbs[0][t][0], wbs[1][t][0], wbs[2][t][0]]
            cur_bs = [wbs[0][t][1], wbs[1][t][1], wbs[2][t][1]]
        wb[t][0] = interpolation(ps, cur_ws, p_cur + step)
        wb[t][1] = interpolation(ps, cur_bs, p_cur + step)
    return wb


def euler(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps)-1):
        dt = time_steps[t+1] - time_steps[t]
        t0 = time_steps[t]
        y0 = y0 + dt*system(t0, y0, params)
        ys.append(y0)
        ys[t+1] = y0
    return ys

def runge_kutta(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps)-1):
        dt = time_steps[t+1]-time_steps[t]
        t0 = time_steps[t]
        t1 = time_steps[t+1]
        k1 = system(t0, y0, params)
        k2 = system(t0/2, y0 + dt / 2 * k1, params)
        k3 = system(t0/2, y0 + dt / 2 * k2, params)
        k4 = system(t1, y0 + dt * k3, params)
        y0  = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y0)
        ys[t+1] = y0
    return ys

def shooting(time_steps, y_approx, system, params, bc, bc_params, solver):   
    eps = 10**(-4)
    t_left = time_steps[len(time_steps)//2::-1]
    t_right = time_steps[len(time_steps)//2:]
    newton_steps = 0
    F = np.zeros(len(y_approx)*len(y_approx)).reshape(len(y_approx), len(y_approx))
    while(True):
        ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
              solver(t_right, y_approx, system, params)[1:]))
        rs = bc(bc_params, ys)
        if (np.abs(rs) < eps).all():
            break
        
        F = np.zeros(len(y_approx)*len(y_approx)).reshape(len(y_approx), len(y_approx))
        for i in range(len(y_approx)):
            yi_approx = y_approx.copy()
            yi_approx[i] += eps
            
            yis = np.concatenate((solver(t_left, yi_approx, system, params)[::-1],
                   solver(t_right, yi_approx, system, params)[1:]))
            rsi = bc(bc_params, yis)
    
            columni = (rsi - rs) / eps
            for j in range(len(F)):
                F[j][i] = columni[j]
        newton_steps += 1
        y_approx =  y_approx - np.dot(inv(F), rs)
    ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                   solver(t_right, y_approx, system, params)[1:]))
    return newton_steps, ys, det(F), svd(F)[1]


def augmented_frechet_matrix(time_steps, y_approx, system, params, cur_hp, bc, bc_params, solver):
    eps = 10**(-4)
    t_left = time_steps[len(time_steps)//2::-1]
    t_right = time_steps[len(time_steps)//2:]
    
    ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                         solver(t_right, y_approx, system, params)[1:]))
    rs = bc(bc_params, ys)
    
    aug_F = np.zeros(len(y_approx)*len(y_approx) + len(y_approx)).reshape(len(y_approx), len(y_approx)+1)
    for i in range(len(y_approx)):
        yi_approx = y_approx.copy()
        yi_approx[i] += eps
        
        yis = np.concatenate((solver(t_left, yi_approx, system, params)[::-1],
               solver(t_right, yi_approx, system, params)[1:]))
        rsi = bc(bc_params, yis)

        columni = (rsi - rs) / eps
        aug_F[:, i] = columni
            
    params[0][cur_hp] += eps
    yps = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
               solver(t_right, y_approx, system, params)[1:]))
    params[0][cur_hp] -= eps
    rsp = bc(bc_params, yps)
    columnp = (rsp - rs) / eps
    aug_F[:, -1] = columnp

    return aug_F

def analyse_point(aug_F):
    dets = []
    for col in range(len(aug_F[0])):
        cols = [i for i in range(len(aug_F[0])) if i != col]
        dets.append(det(aug_F[:, cols]))
    return np.round(dets, 4)


    
    
    
    