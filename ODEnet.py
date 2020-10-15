import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import euler, shooting, runge_kutta, augmented_frechet_matrix, analyse_point


class ODEnet:
    
    def __init__(self, hyper_parameters, structure, time_step = 0.1):
        self.HP = hyper_parameters
        self.ts = time_step
        
        self.structure = structure
        self.wb = self.init_weights()
    
        self.Z = []
        self.A = []
    
    def init_weights(self):
        wb = []
        for t in range(self.r_f2i(self.HP['T']*(1/self.ts))):
            wb.append([])
            wb[t].append(np.zeros((self.structure[t+1], self.structure[t])))
            wb[t].append(np.zeros(self.structure[t+1]))
        wb.append([])
        wb[-1].append(np.zeros((self.structure[-1], self.structure[-1])))
        wb[-1].append(np.zeros(self.structure[-1]))
        return wb
   
    def update_parameter(self, p, step):
        self.HP[p] += step
        
    def update_weights(self, wb):
        self.wb = wb.copy()
    def reset_weights(self):
        self.wb = self.init_weights()

    def f(self, t, z0, wb):
        e1, e2 = self.HP['E1'], self.HP['E2']
        t = self.t2t(t)
        w, b = wb[t][0], wb[t][1]
        dz = e1*self.activation(np.dot(w, z0) + b) - e2*z0
        return dz
    
    def a_adjoint(self, t, a0, parameters):
        e1, e2 = self.HP['E1'], self.HP['E2']
        t = self.t2t(t)
        z, w, b = parameters[0][t], parameters[1][t][0], parameters[1][t][1]
        da = -np.dot(a0, (w*e1*self.der_activation(np.dot(w, z) + b).reshape(len(z), -1) - e2*np.ones(len(z)).reshape(len(z), -1)))
        return da
    
    def w_adjoint(self, t, dw0, parameters):
        t = self.t2t(t)
        z, a, w, b = parameters[0][t], parameters[1][t], parameters[2][t][0], parameters[2][t][1]
        der_f = self.der_activation(np.dot(w, z) + b)
        db = -(a*der_f).flatten()
        der_f = np.array([der_f for _ in range(len(der_f))]).T * z
        dw = -(a.reshape(-1, 1)*der_f).flatten()
        return np.array([*dw, *db])
    
    # def augmented_f(self, t, za0, wb):
    #     e1, e2 = self.HP['E1'], self.HP['E2']
    #     t = self.t2t(t)
    #     w, b = wb[t][0], wb[t][1]
    #     z0, a0 = za0[:len(za0)//2], za0[len(za0)//2:]
    #     dz = e1*self.activation(np.dot(w, z0) + b) - e2*z0
    #     da = -np.dot(a0, (w*e1*self.der_activation(np.dot(w, z0) + b).reshape(len(z0), -1) - e2*np.ones(len(z0)).reshape(len(z0), -1)))
    #     return np.array([*dz, *da])
        
    def augmented_f(self, t, za0, params):
        hp, wb = params[0], params[1]
        e1, e2 = hp['E1'],hp['E2']
        t = self.t2t(t)
        w, b = wb[t][0], wb[t][1]
        z0, a0 = za0[:len(za0)//2], za0[len(za0)//2:]
        dz = e1*self.activation(np.dot(w, z0) + b) - e2*z0
        da = -np.dot(a0, (w*e1*self.der_activation(np.dot(w, z0) + b).reshape(len(z0), -1) - e2*np.ones(len(z0)).reshape(len(z0), -1)))
        return np.array([*dz, *da])
    
    def bc(self, bc_params, za):
        z0_true, z1_true = bc_params[0], bc_params[1]
        z0 = za[0][:len(za[0])//2]
        z1 = za[-1][:len(za[-1])//2]
        a1 = za[-1][len(za[-1])//2:]
        
        z0 = z0 - z0_true
        a1 = a1 - z1 + z1_true
        return np.concatenate((z0, a1))
    
    # def activation(self, x):
    #     return x
    # def der_activation(self, x):
    #     return np.ones_like(x)
    def activation(self, x):
        return np.tanh(x)
    def der_activation(self, x):
        return np.ones(len(x)) - np.tanh(x)**2

    def time_steps(self, t0, t1):
        T = []
        if t0 > t1:
            while t0 > t1:
                T.append(t0)
                t0 = np.round(t0 - self.ts, 1)
            T.append(t0)
        else:
            while t0 < t1:
                T.append(t0)
                t0 = np.round(t0 + self.ts, 1)
            T.append(t0)
        return T
            
    def r_f2i(self, x):
        return int(np.round(x))
    def t2t(self, x):
        return int(np.round(x/self.ts))
    def change_weights(self, dwb):
        dwb.reverse()
        for t in range(len(self.wb)):
            
            dw = np.array(dwb[t][:-len(self.wb[t][1])])
            db = np.array(dwb[t][-len(self.wb[t][1]):])
            dw = dw.reshape(self.wb[t][0].shape)
            db = db.reshape(self.wb[t][1].shape)
            self.wb[t][0] -= self.HP['LR']*dw
            self.wb[t][1] -= self.HP['LR']*db
            
    def forward(self, z0):
        time_steps = self.time_steps(0, self.HP['T'])
        self.Z = euler(time_steps, z0, self.f, self.wb)

    def back_propagation(self, zs):
        a0 = -self.Z[-1] + zs
        time_steps = self.time_steps(self.HP['T'], 0)
        time_steps.reverse()
        self.A = euler(time_steps, a0, self.a_adjoint, (self.Z, self.wb))
        dwb0 = np.zeros(len(zs)**2 + len(zs))
        dwb = euler(time_steps, dwb0, self.w_adjoint, (self.Z, self.A, self.wb))
        self.change_weights(dwb)
    
    def augmented_propagation(self, za0, z0, zs):
        time_steps = self.time_steps(0, self.HP['T'])
        newton_steps, za, det, sv = shooting(time_steps, za0, self.augmented_f, [self.HP, self.wb], self.bc, [z0, zs], runge_kutta)
        self.save_za(za)
        time_steps.reverse()
        dwb0 = np.zeros(len(zs)**2 + len(zs))
        dwb = euler(time_steps, dwb0, self.w_adjoint, (self.Z, self.A, self.wb))
        self.change_weights(dwb)
        return newton_steps, det, sv
    
    def matrix_analyses(self, za0, z0, zs):
        time_steps = self.time_steps(0, self.HP['T'])
        aug_F = augmented_frechet_matrix(time_steps, za0, self.augmented_f, [self.HP, self.wb], 'E2', self.bc, [z0, zs], runge_kutta)
        with open("dets.txt", 'a') as f:
            print(analyse_point(aug_F), file=f)
        
    def save_za(self, za):
        z = []
        a = []
        for l in za:
            z.append(l[:len(l)//2])
            a.append(l[len(l)//2:])
        self.Z = z
        self.A = a
    
    def plot_z(self, z0, zs, epoch, parameter=False, determinant=False):
        time_steps = self.time_steps(0, self.HP['T'])
        z = np.array(self.Z).T
        colors = ['b', 'g', 'r', 'm', 'c', 'silver', 'y', 'lightcoral', 'lime', 'gold']
        for i in range(len(zs)):
            plt.scatter(time_steps[0], z0[i], c=colors[i], marker='o')
            plt.scatter(time_steps[-1], zs[i], c=colors[i], marker='o')
            plt.plot(time_steps, z[i], c=colors[i])
            if parameter and determinant:
                plt.title(f"epoch {epoch} : {parameter} = {np.round(self.HP[parameter], 4)} : determinant = {np.round(determinant, 4)}")
            else:
                plt.title(f"epoch {epoch}")
        plt.ylim(-1.1, 1.1)
        plt.show()
    