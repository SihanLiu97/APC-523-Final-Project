import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import pyfftw
import os
import time
import sys
np.random.seed(1212)
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 12

N_component = 4
ratio_ls=[1, 1, 1, 1]
x = 1
y = 1
dx = 1 / 128
N_x = int(x / dx)
N_y = int(y / dx)
lambda_ = 2.0e-5
dt = 0.5 * lambda_

chi = np.zeros((N_component, N_component), dtype=np.float32)

chi_12 = sys.argv[1]
chi_13 = sys.argv[2]
chi_14 = sys.argv[3]
chi_23 = sys.argv[4]
chi_24 = sys.argv[5]
chi_34 = sys.argv[6]

chi[0, 1] = chi[1, 0] = chi_12
chi[0, 2] = chi[2, 0] = chi_13
chi[0, 3] = chi[3, 0] = chi_14
chi[1, 2] = chi[2, 1] = chi_23
chi[1, 3] = chi[3, 1] = chi_24
chi[2, 3] = chi[3, 2] = chi_34
# print(chi)
'''
fft_a = pyfftw.empty_aligned((N_component, N_y, N_x), dtype='complex64')
fft_b = pyfftw.empty_aligned((N_component, N_y, N_x), dtype='complex64')
fft_object = pyfftw.FFTW(fft_a, fft_b, axes=(1,2))

ifft_a = pyfftw.empty_aligned((N_component, N_y, N_x), dtype='complex64')
ifft_b = pyfftw.empty_aligned((N_component, N_y, N_x), dtype='complex64')
ifft_object = pyfftw.FFTW(ifft_a, ifft_b, direction='FFTW_BACKWARD', axes=(1,2))

def fft(x):
    fft_a[:] = x
    return fft_object().copy()


def ifft(x):
    ifft_a[:] = x
    return ifft_object().copy()
'''

class PhaseField2D:
    def __init__(self, x, y, dx, N_component, ratio_ls, chi, lambda_, dt):
        self.x = x
        self.y = y
        self.dx = dx
        self.N_x = int(x / dx)
        self.N_y = int(y / dx)
        self.N_component = N_component
        self.phi = np.empty((N_component, N_y, N_x), dtype=np.float32)
        self.phi[:] = np.array(ratio_ls)[:, np.newaxis, np.newaxis] / np.sum(ratio_ls)  # phi_xyi = ratio_i
        # print(self.phi)

        self.chi = chi
        self.lambda_ = lambda_
        self.dt = dt
        self.A = 0.5 * chi.max() * lambda_

        k_x  = np.float32(2 * np.pi * np.fft.fftfreq(N_x, self.dx))
        k_y  = np.float32(2 * np.pi * np.fft.fftfreq(N_y, self.dx))
        self.k_x, self.k_y = np.meshgrid(k_x, k_y)
        self.k2 = self.k_x**2 + self.k_y**2
        self.k4 = self.k_x**4 + self.k_y**4
        self.ik_x = self.k_x * 1j
        self.ik_y = self.k_y * 1j

    def add_noise(self, epsilon):
        noise = np.random.uniform(-epsilon, epsilon, self.phi.shape)
        noise -= noise.mean(0)  # sum_i noise_xyi = 0
        noise -= noise.mean((1, 2)).reshape(self.N_component, 1, 1)  # sum_xy noise_xyi = 0
        self.phi += noise
        # print(self.phi.mean((1, 2)))
        # print(self.phi.sum(0))
        # print(self.phi)
    
    def write_phi(self, phi, output_dir, filename):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        '''
        with open(f'{output_dir}/{filename}.txt', 'w') as f:
            for i in range(self.N_component):
                f.write(f'phi_{i + 1}\n')
                for row in phi[i]:
                    row_str = ' '.join(f'{p:12.7f}' for p in row)
                    f.write(row_str + '\n')
                f.write('\n')
            f.write(f'sum_i phi_xyi\n')
            for row in phi.sum(0):
                row_str = ' '.join(f'{p:12.7f}' for p in row)
                f.write(row_str + '\n')
            f.write('\nsum_xy phi_xyi\n')
            for row in phi.mean((1, 2)):
                f.write(str(row) + '\n')
        '''
        np.save(f'{output_dir}/{filename}.npy', phi)

    def visualize_N34(self, phi, output_dir, step, method=''):
        if 3 <= self.N_component <= 4:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.figure(figsize=(5,5))
            plt.gca().set_aspect('equal')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.title(f'{method}, Step = {step}')
            image = np.swapaxes(phi, 0, 2)
            if self.N_component == 3:
                plt.imshow(image, interpolation='bicubic', extent=(0, self.x, 0, self.y))
            if self.N_component == 4:
                a = image.copy()  # must use copy!
                a[..., -1] = 1 - a[..., -1]
                plt.imshow(a, interpolation='bicubic', extent=(0, self.x, 0, self.y))
            plt.savefig(f'{output_dir}/{step}.png', dpi=220)
            plt.close()
    
    def free_energy(self, phi):
        phi_temp = phi.copy()
        phi_temp[phi_temp <= 0.0] = 1e-10
        entropy_term = np.sum(phi_temp * np.log(phi_temp), axis=0)
        chi_phi = np.einsum('ij,jyx->iyx', self.chi, phi)
        energy_term = 0.5 * np.einsum('iyx,iyx->yx', chi_phi, phi)
        grad_phi_x = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / 2 / self.dx
        grad_phi_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2 / self.dx
        chi_grad_phi_x = np.einsum('ij,jyx->iyx', self.chi, grad_phi_x)
        chi_grad_phi_grad_phi_x = np.einsum('iyx,iyx->yx', chi_grad_phi_x, grad_phi_x)
        chi_grad_phi_y = np.einsum('ij,jyx->iyx', self.chi, grad_phi_y)
        chi_grad_phi_grad_phi_y = np.einsum('iyx,iyx->yx', chi_grad_phi_y, grad_phi_y)
        interface_term = - self.lambda_ / 2 * (chi_grad_phi_grad_phi_x + chi_grad_phi_grad_phi_y)
        free_energy = entropy_term + energy_term + interface_term
        return free_energy.sum()
    
    def mu(self, phi):
        laplacian_phi = (np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1) +
                          np.roll(phi, -1, axis=2) + np.roll(phi, 1, axis=2) - 4 * phi) / self.dx**2
        mu = np.einsum('ij,jyx->iyx', self.chi, phi + lambda_ * laplacian_phi)
        return mu

    def mu_tilde(self, phi_tilde):
        return np.einsum('ij,jyx->iyx', self.chi, (1.0 - self.k2 * self.lambda_) * phi_tilde)

    def grad_mu(self, phi):
        mu = self.mu(phi)
        grad_mu_x = (np.roll(mu, -1, axis=2) - np.roll(mu, 1, axis=2)) / 2 / self.dx
        grad_mu_y = (np.roll(mu, -1, axis=1) - np.roll(mu, 1, axis=1)) / 2 / self.dx
        return grad_mu_x, grad_mu_y

    def J_i(self, phi, grad_mu_i):
        J_0 = np.einsum('iyx,iyx->yx', phi, grad_mu_i)
        return np.einsum('iyx,iyx->iyx', phi, grad_mu_i - J_0)
    
    def grad_J(self, phi):
        grad_mu_x, grad_mu_y = self.grad_mu(phi)
        J_x = self.J_i(phi, grad_mu_x)
        J_y = self.J_i(phi, grad_mu_y)
        grad_J_entropy = (np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1) +
                          np.roll(phi, -1, axis=2) + np.roll(phi, 1, axis=2) - 4 * phi) / self.dx**2
        grad_J_energy = (np.roll(J_x, -1, axis=2) - np.roll(J_x, 1, axis=2)) / 2 / self.dx + \
                        (np.roll(J_y, -1, axis=1) - np.roll(J_y, 1, axis=1)) / 2 / self.dx
        return grad_J_entropy + grad_J_energy

    def N_tilde(self, phi, phi_tilde):
        N_entropy = - self.k2 * phi_tilde
        N_4th = self.A * self.k4 * phi_tilde
        # mu_tilde = self.mu_tilde(phi_tilde)
        # grad_mu_x = ifft(self.ik_x * mu_tilde)
        # grad_mu_y = ifft(self.ik_y * mu_tilde)
        grad_mu_x, grad_mu_y = self.grad_mu(phi)
        J_x = self.J_i(phi, grad_mu_x)
        J_y = self.J_i(phi, grad_mu_y)
        N_energy = self.ik_x * fft(J_x) + self.ik_y * fft(J_y)
        return N_entropy + N_energy + N_4th
    
    def solve_FFT(self, step_end, step_every, step_start=0, output_dir='FFT_data'):
        step_now = step_start
        if step_start > 0:
            self.phi = np.load(f'{output_dir}/{step_start}.npy')
        self.write_phi(self.phi, output_dir, step_now)
        self.visualize_N34(self.phi, output_dir, step_now, method='FFT')
        with open(f'{output_dir}/free_energy.txt', mode='a') as f:
            f.write(f'{step_now:7} {self.free_energy(self.phi):14.7f} {self.phi.min():14.7f} {self.phi.max():14.7f} {self.phi.mean((1, 2)).sum():14.7f} ')
            for row in self.phi.mean((1, 2)):
                f.write(f'{row:14.7f}')
            f.write('\n')
        phi_tilde = fft(self.phi)
        start_time = time.time()
        while step_now < step_end:
            step_now += 1
            phi_n_tilde = phi_tilde
            N_tilde = self.N_tilde(self.phi, phi_n_tilde)
            phi_tilde = (phi_n_tilde + self.dt * N_tilde) / (1 + self.A * self.k4 * self.dt)
            self.phi = ifft(phi_tilde).real
            if (step_now % step_every) == 0:
                # self.phi[self.phi < 0.0] = 0.0
                if np.isnan(self.phi.max()):
                    print('NAN encountered')
                    break
                with open(f'{output_dir}/free_energy.txt', mode='a') as f:
                    f.write(f'{step_now:7} {self.free_energy(self.phi):14.7f} {self.phi.min():14.7f} {self.phi.max():14.7f} {self.phi.mean((1, 2)).sum():14.7f} ')
                    for row in self.phi.mean((1, 2)):
                        f.write(f'{row:14.7f}')
                    f.write('\n')
                self.write_phi(self.phi, output_dir, step_now)
                self.visualize_N34(self.phi, output_dir, step_now, method='FFT')
                print(f'Step = {step_now}. Performance: {(step_now - step_start) / (time.time() - start_time):.5f} steps/s')
                print(f'phi_min = {self.phi.min():.7f}, phi_max = {self.phi.max():.7f}')
                print(f'free energy = {self.free_energy(self.phi):.7f}\n')
    
    def solve_RK4(self, step_end, step_every, step_start=0, output_dir='RK4_data'):
        step_now = step_start
        if step_start > 0:
            self.phi = np.load(f'{output_dir}/{step_start}.npy')
        self.write_phi(self.phi, output_dir, step_now)
        self.visualize_N34(self.phi, output_dir, step_now, method='RK4')
        with open(f'{output_dir}/free_energy.txt', mode='a') as f:
            f.write(f'{step_now:7} {self.free_energy(self.phi):14.7f} {self.phi.min():14.7f} {self.phi.max():14.7f} {self.phi.mean((1, 2)).sum():14.7f} ')
            for row in self.phi.mean((1, 2)):
                f.write(f'{row:14.7f}')
            f.write('\n')
        start_time = time.time()
        while step_now < step_end:
            step_now += 1
            k1 = self.grad_J(self.phi)
            phi_temp = self.phi + 0.5 * self.dt * k1
            k2 = self.grad_J(phi_temp)
            phi_temp = self.phi + 0.5 * self.dt * k2
            k3 = self.grad_J(phi_temp)
            phi_temp = self.phi + self.dt * k3
            k4 = self.grad_J(phi_temp)
            self.phi = self.phi + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (step_now % step_every) == 0:
                # self.phi[self.phi < 0.0] = 0.0
                if np.isnan(self.phi.max()):
                    print('NAN encountered')
                    break
                with open(f'{output_dir}/free_energy.txt', mode='a') as f:
                    f.write(f'{step_now:7} {self.free_energy(self.phi):14.7f} {self.phi.min():14.7f} {self.phi.max():14.7f} {self.phi.mean((1, 2)).sum():14.7f} ')
                    for row in self.phi.mean((1, 2)):
                        f.write(f'{row:14.7f}')
                    f.write('\n')
                self.write_phi(self.phi, output_dir, step_now)
                self.visualize_N34(self.phi, output_dir, step_now, method='RK4')
                print(f'Step = {step_now}. Performance: {(step_now - step_start) / (time.time() - start_time):.5f} steps/s')
                print(f'phi_min = {self.phi.min():.7f}, phi_max = {self.phi.max():.7f}')
                print(f'free energy = {self.free_energy(self.phi):.7f}\n')


field = PhaseField2D(x, y, dx, N_component, ratio_ls, chi, lambda_, dt)
field.add_noise(0.001)
# field.solve_FFT(step_end=1000000, step_every=1000, step_start=0, output_dir=f'FFT_data_{chi_12}_{chi_13}_{chi_14}_{chi_23}_{chi_24}_{chi_34}')
field.solve_RK4(step_end=1000000, step_every=1000, step_start=0, output_dir=f'RK4_data_{chi_12}_{chi_13}_{chi_14}_{chi_23}_{chi_24}_{chi_34}')
