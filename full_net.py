import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

class ODEnet:
    def __init__(self, shape, eta1, eta2, LR):
        self.shape = shape
        self.eta1 = eta1
        self.eta2 = eta2
        self.LR = LR
        
        self.OLDETA2 = eta2
        self.REVERSE = False
        
        self.BIF = False
        
        self.try_count = 0
        
        self.eps = 10**(-4)
        self.S = self.shape[0]*2
        
        self.iters = 0
        self.init_step = [-0.01, -0.01, -0.01, -0.01, -0.1]
        self.min_step = [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
        self.step = self.init_step[-1]
        T = np.round(len(shape)/10, 1)
        self.time_steps = np.linspace(0, T, len(shape))
        
        self.w = self.gen_weights()
        self.ZA = []
        
        # some random initialisation
        self.init_v = np.random.uniform(-0.5, 0.5, self.shape[0]*2)
        self.z_zero = np.random.uniform(-1, 1, self.shape[0])
        self.z_star = np.random.uniform(-1, 1, self.shape[0])
        
        self.Jacob_Matrix = np.zeros(len(self.init_v)**2).reshape(len(self.init_v),len(self.init_v))
        
        # init fix param s.t. not to be chosen initially
        self.fix_param = self.S

        # some params to keep track of algorithm
        self.norm = 0
        self.D = 0
        self.SV = []
        
    def activation(self, x):
        return np.tanh(x)
    def der_activation(self, x):
        return 1 - np.tanh(x)**2

    def gen_weights(self):
        W = []
        rows = [[0.01*(-1)**i for i in range(self.shape[0])], [0.01*(-1)**i for i in range(1, self.shape[0]+1)]]
        for i in range(len(self.time_steps)-1):
            if i % 2 == 0:
                W.append([np.array([rows[i%2] for i in range(self.shape[0])]), np.array([0.02*(-1)**i for i in range(self.shape[0])])])
            else:
                W.append([np.array([rows[i%2] for i in range(1, self.shape[0]+1)]), np.array([0.02*(-1)**i for i in range(1, self.shape[0]+1)])])
        return W
    
    def odesolveF(self, y0, w, step):
        k1 = self.dynamic_system_net(y0, w)
        k2 = self.dynamic_system_net(y0+step*k1/2, w)
        k3 = self.dynamic_system_net(y0 + step*k2/2, w)
        k4 = self.dynamic_system_net(y0 + step*k3, w)
        return y0 + 1/6 * step * (k1 + 2*k2 + 2*k3 + k4)
        
    def dynamic_system_net(self, za_cur, w):
    
        Z = za_cur[:len(za_cur)//2]
        A = za_cur[len(za_cur)//2:]
        dzdt = self.eta1*self.activation(np.dot(Z, w[0]) + w[1]) - self.eta2*Z
    
        zeros = np.zeros(len(Z) - len(dzdt))
    
        idt = np.ones(len(Z)) * self.eta2
        dadt = -np.dot(((self.der_activation(np.dot(Z, w[0]) + w[1]) * self.eta1) * w[0] - idt.T), A)
        
        dzadt = np.concatenate((dzdt, zeros, dadt, zeros), axis=None)
        return dzadt
    
    
    def solve_forward(self, init_v, p_shift):
        self.eta2 += self.eps * p_shift
        
        t_half = len(self.time_steps)//2
        ZA = np.array([np.empty_like(self.time_steps) for _ in range(len(init_v))])
        for i in range(len(ZA)):
            ZA[i][t_half] = init_v[i]
    
        for i in range(t_half-1, -1, -1):
            tstep = self.time_steps[i] - self.time_steps[i+1]
            
            za_i = self.odesolveF(init_v, self.w[i], tstep)
            for j in range(len(za_i)):
                ZA[j][i] = za_i[j]
            init_v = za_i
            
        init_v = np.empty(len(init_v))
        for i in range(len(init_v)):
            init_v[i] = ZA[i][t_half]
    
        for i in range(t_half+1, len(self.time_steps)):
            tstep = self.time_steps[i] - self.time_steps[i-1]
            
            za_i = self.odesolveF(init_v, self.w[i-1], tstep)
    
            for j in range(len(za_i)):
                ZA[j][i] = za_i[j]
            init_v = za_i
        
        for i in range(len(init_v)):
            init_v[i] = ZA[i][t_half]
            
        self.eta2 -= self.eps * p_shift
        return ZA
    
    
    def find_discrepancies(self, ZA, ideal):
        discrepancies = np.empty(len(ZA))
        for i in range(len(ZA)//2):
            discrepancies[i] = ZA[i][0]
        for i in range(len(ZA)//2, len(ZA)):
            discrepancies[i] = ZA[i][-1]
        return ideal-discrepancies

    def find_ideal(self, ZA):
        ideal = np.empty(len(ZA))
        for i in range(len(ideal)//2):
            ideal[i] = self.z_zero[i]
        for i in range(len(ideal)//2, len(ideal)):
            ideal[i] = self.z_star[i-len(ideal)//2] - ZA[i-len(ideal)//2][-1]
        return ideal
    
    def find_i_der(self, discrepancies, discrepancies_i, i):
        i_der = (discrepancies_i - discrepancies) / self.eps
        for j in range(len(self.Jacob_Matrix)):
            self.Jacob_Matrix[j][i] = i_der[j]
    
    def find_jacob_matrix(self, discrepancies, ideal):
    
        for i in range(len(self.init_v)):
            init_vi = self.init_v.copy()
            if i == self.fix_param:
                ZAi = self.solve_forward(init_vi, 1)
            else:
                init_vi[i] += self.eps
                ZAi = self.solve_forward(init_vi, 0)
            ideal_i = self.find_ideal(ZAi)
            discrepancies_i = self.find_discrepancies(ZAi, ideal_i)
            self.find_i_der(discrepancies, discrepancies_i, i)
    
    def newton(self, v, discrepancies):
        new_v =  v - np.dot(np.linalg.inv(self.Jacob_Matrix), discrepancies)
        return new_v

    def check(self, discrepancies):
        for i in range(len(discrepancies)):
            if np.abs(discrepancies[i]) > self.eps:
                return False
        return True
    
    
    def forward(self):
        self.iters = 0
        while True:
            self.iters += 1
            if self.iters >= 10:
                break
            ZA = self.solve_forward(self.init_v, 0)

            ideal = self.find_ideal(ZA)
            discrepancies = self.find_discrepancies(ZA, ideal)
            self.find_jacob_matrix(discrepancies, ideal)
            if np.linalg.det(self.Jacob_Matrix) == 0:
                self.iters = 7
                break
            
            v = self.get_cur_v()
            v = self.newton(v, discrepancies)
            
            for i in range(len(v)):
                if i == self.fix_param:
                    self.eta2 = v[i]
                else:
                    self.init_v[i] = v[i]
            if self.check(discrepancies):
                
                ZA = self.solve_forward(self.init_v, 0)
                ideal = self.find_ideal(ZA)
                discrepancies = self.find_discrepancies(ZA, ideal)
                
                self.find_jacob_matrix(discrepancies, ideal)
                
                old_param = self.fix_param
                self.fix_param = self.S
                
                self.find_jacob_matrix(discrepancies, ideal)
                self.D = np.linalg.det(self.Jacob_Matrix)
                if self.D == 0:
                    self.BIF =True
                self.SV = np.linalg.svd(self.Jacob_Matrix)[1]
                self.fix_param = old_param
                self.ZA = ZA
                break
       
            
    def odesolveB(self, w0, w, y, y_, step):
        k1 = self.back_prop_system(w0, w, y, y_)
        k2 = self.back_prop_system(w0+step*k1/2, w, y, y_)
        k3 = self.back_prop_system(w0 + step*k2/2, w, y, y_)
        k4 = self.back_prop_system(w0 + step*k3, w, y, y_)
        return w0 + 1/6 * step * (k1 + 2*k2 + 2*k3 + k4)   
    
            
    def back_prop_system(self, w0, w, za_cur, za_cur_1):
    
        Z = za_cur_1[:len(za_cur_1)//2]
        A = za_cur[len(za_cur)//2:]
        
        der_f = self.der_activation(np.dot(Z, w[0]) + w[1])
        
        Z = Z.reshape(len(Z), 1)
    
        dldw = np.concatenate([der_f for i in range(self.S//2)]).reshape(len(der_f), len(der_f))
        dldw = self.eta1 * Z * (A * dldw)
        
        dldb = der_f * A * self.eta1
        
        return np.concatenate((dldw.reshape(1, -1)[0], dldb))

    def back_prop(self):
        t = len(self.time_steps)
        dldw0 = np.zeros(6)
    
        for i in range(t-1, 0, -1):
            za_cur = np.empty(len(self.ZA))
            za_cur_1 = np.empty(len(self.ZA))
            for j in range(len(self.ZA)):
                za_cur[j] = self.ZA[j][i]
                za_cur_1[j] = self.ZA[j][i-1]
    
            tstep = self.time_steps[i-1] - self.time_steps[i]
            dldw0 = self.odesolveB(dldw0, self.w[i-1], za_cur, za_cur_1, tstep)
            
            self.update_weights(i-1, dldw0)
    
        
    def update_weights(self, i, dldw):
        self.w[i][0] -= self.LR*dldw[0:self.shape[i]*self.shape[i+1]].reshape(self.w[i][0].shape)
        self.w[i][1] -= self.LR*dldw[self.shape[i]*self.shape[i+1]:]
    
    def interpolation(self, f, x, x_next):
        a0 = f[0]
        if len(f) == 1:
            return a0
        
        a1 = (f[1] - f[0])/(x[1]-x[0])
        if len(f) == 2:
            return a0 + a1*(x_next-x[0])
        
        fx1x2 = (f[2]-f[1])/(x[2]-x[1])
        a2 = (fx1x2 - a1)/(x[2]-x[0])
        
        if len(f) == 3:
            fx1x2 = (f[2]-f[1])/(x[2]-x[1])
            a2 = (fx1x2 - a1)/(x[2]-x[0])
        return a0 + a1*(x_next-x[0]) + a2*(x_next-x[0])*(x_next-x[1])
    
    
    def update_arrs(self, arr_v, arr_p, new_v, new_p):
        if len(arr_v) == 3:
            arr_v[0], arr_v[1], arr_v[2] = arr_v[1], arr_v[2], new_v
            arr_p[0], arr_p[1], arr_p[2] = arr_p[1], arr_p[2], new_p
        else:
            arr_v.append(new_v)
            arr_p.append(new_p)
    
    def get_cur_v(self):
        v = np.zeros(len(self.init_v))
        for i in range(len(self.init_v)):
            if i == self.fix_param:
                v[i] = self.eta2
            else:
                v[i] = self.init_v[i]
        return v
    
    
    def change_polinom(self, arr_v, arr_p):
        if self.fix_param != self.S:
            arr_v[0][self.fix_param], arr_p[0] = arr_p[0], arr_v[0][self.fix_param]
            arr_v[1][self.fix_param], arr_p[1] = arr_p[1], arr_v[1][self.fix_param]
            arr_v[2][self.fix_param], arr_p[2] = arr_p[2], arr_v[2][self.fix_param]
    
    def reverse_polinom(self, arr_v, arr_p):
        arr_v[0], arr_v[1], arr_v[2] = arr_v[2], arr_v[1], arr_v[0]
        arr_p[0], arr_p[1], arr_p[2] = arr_p[2], arr_p[1], arr_p[0]
        self.REVERSE = not self.REVERSE
    
    
    def find_p_fix(self):
        if self.fix_param == self.S:
            v1 = self.init_v.copy()
            self.eta2 += self.eps
            self.forward()
            v2 = self.init_v.copy()
            
            self.eta2 -= self.eps
            self.init_v = v1
            
            self.fix_param = np.argmax(np.abs(v2-v1))

        else:
            v1 = self.get_cur_v()
            old_eta2 = self.eta2
            old_v = self.init_v.copy()
            self.init_v[self.fix_param] += self.eps
            self.forward()
            v2 = self.get_cur_v()
            
            self.eta2 = old_eta2
            self.init_v = old_v

            
            n = np.argmax(np.abs(v2-v1))

            if n == self.fix_param:
                self.fix_param = self.S
            else:
                self.fix_param = n

    def check_D(self, M):
        if np.linalg.det(M) == 0:
            self.iters = 7
            return True
        else:
            return False
        
    def check_BIF(self):
        if self.BIF:
            self.BIF = False
            
            if self.fix_param == self.S:
                self.eta2 -= self.step
                self.step = self.step * 10
                self.eta2  += self.step
            else:
                self.init_v[self.fix_param] -= self.step
                self.step = 10*self.step
                self.init_v[self.fix_param] += self.step
        
    
    def check_iter(self, old_v, old_eta2, arr_v, arr_p):
        if self.iters >= 7:
            if self.fix_param == self.S:
                self.eta2 -= self.step
                self.init_v = old_v
                
                self.step /= 2
            else:
                self.eta2 = old_eta2
                self.init_v = old_v
                self.init_v[self.fix_param] -= self.step
                self.step /= 2
                
            if np.abs(self.step)< self.min_step[self.fix_param]:
                if self.OLDETA2 == self.eta2 and self.REVERSE == False:
                    self.step = -self.init_step[self.fix_param]
                    self.reverse_polinom(arr_v, arr_p)
                    
                else:
                    if self.REVERSE:
                        self.reverse_polinom(arr_v, arr_p)
                        self.try_count += 1
                    
                    self.change_polinom(arr_v, arr_p)
                    self.find_p_fix()
                    self.OLDETA2 = self.eta2
                    
                    self.change_polinom(arr_v, arr_p)
                    
                    self.step = self.init_step[self.fix_param]
                    
                    if self.fix_param == self.S:
                        self.eta2 += self.step
                    else:
                        self.init_v[self.fix_param] += self.step
            else:
                if self.fix_param == self.S:
                    self.eta2 += self.step
                else:
                    self.init_v[self.fix_param] += self.step
            return True
        else:
            if self.iters < 3:
                self.step *= 2
                if np.abs(self.step) > 0.1 and self.fix_param == self.S:
                    self.step = -0.1
            return False
            
                
    def model_info(self):
        print(f"STEP: {self.step}")
        print(f"ETA1: {self.eta1}")
        print(f"ETA2: {self.eta2}")
        print(f"LR: {self.LR}")
        print(f"FP: {self.fix_param}")

    def trained(self):
        print(self.ZA)
        print(self.S)
        z1 = []
        for i in range(self.S//2):
            z1.append(self.ZA[i][0])
        for i in range(self.S//2):
            z1.append(self.ZA[i][-1])
        z2 = np.array([*self.z_zero, *self.z_star])
        acc = np.abs(z2-z1)
        print(f"ACC: {np.round(acc, 10)}")
        for a in acc:
            if a > 10**(-2):
                return False
        return True
    
    def restore(self, arr_v, arr_p):
        if self.REVERSE:
            self.reverse(arr_v, arr_p)
            self.REVERSE = False
        self.change_polinom(arr_v, arr_p)
        self.fix_param = self.S
        self.w = self.gen_weights()
        self.step = self.init_step[self.fix_param]
    
    
    def move_along_P(self, arr_v, arr_p):
        if self.fix_param == self.S:
            self.update_arrs(arr_v, arr_p, self.init_v, self.eta2)
            self.init_v = self.interpolation(arr_v, arr_p, arr_p[-1]+self.step)
            self.eta2 += self.step
        else:
            v = self.get_cur_v()
            self.update_arrs(arr_v, arr_p, v, self.init_v[self.fix_param])
            v = self.interpolation(arr_v, arr_p, arr_p[-1]+self.step)
            for i in range(len(self.init_v)):
                if i == self.fix_param:
                    self.init_v[i] += self.step
                else:
                    self.init_v[i] = v[i]
            self.eta2 = v[self.fix_param]
        self.model_info()
    
    
    
    def create_csv(self, file):
        fieldsnames = ['E2', 'D', 'S4', 'S3', 'S2', 'S1']
        with open(file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldsnames)
            writer.writeheader()
            
    def write_csv(self, file):
        fieldsnames = ['E2', 'D', 'S4', 'S3', 'S2', 'S1']
        with open(file, 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldsnames)
            
            info = {
                    'E2': self.eta2,
                    'D': self.D,
                    'S4': self.SV[-4],
                    'S3': self.SV[-3],
                    'S2': self.SV[-2],
                    'S1': self.SV[-1],
                    
                    }
            
            csv_writer.writerow(info)
        
        
    def conduct_experiment(self, file):
        arr_v = []
        arr_p = []
        
        self.create_csv(file)
        
        # self.model_info()
        i = 0
        all_iters = 0

        while True:
            i += 1
            if all_iters > 10000000:
                break
            print(i)
            all_iters+=1
            print("eta2 : ", self.eta2)
            print("step : ", self.step)
            print("fixp : ", self.fix_param)
            print("init : ", self.init_v)
            print("dett : ", self.D)
            old_v = self.init_v.copy()
            old_eta2 = self.eta2
            self.forward()
            if all_iters == 1 or all_iters == 1000:

            if self.check_iter(old_v, old_eta2, arr_v, arr_p): continue
            if not self.trained():
                self.back_prop()
                continue
            break
            i = 0
            self.write_csv(file)
            self.move_along_P(arr_v, arr_p) 
            self.restore(arr_v, arr_p)
            

# defining hyperparameters
n = 2
E1 = 1
E2 = -0.1
LR = 0.1
for t in range(20, 211):
    nn = ODEnet(np.ones(t, dtype=np.int)*n, E1, E2, LR)
    file = str(n)+'N.T=' + str(np.round(t/10, 2)) + ',e1=1,lr=0.5.csv'
    nn.conduct_experiment(file)
    break
