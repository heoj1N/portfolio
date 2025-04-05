import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Enable LaTeX-style plotting
plt.style.use('seaborn-darkgrid')
rc('text', usetex=True)
rc('font', family='serif')

# Generate data points
x = np.linspace(-3, 3, 200)

# Create subplots for different basis functions
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Common Basis Functions in Machine Learning', fontsize=16)

# 1. Polynomial basis functions
ax1.plot(x, np.ones_like(x), label=r'$\phi_0(x) = 1$')
ax1.plot(x, x, label=r'$\phi_1(x) = x$')
ax1.plot(x, x**2, label=r'$\phi_2(x) = x^2$')
ax1.plot(x, x**3, label=r'$\phi_3(x) = x^3$')
ax1.set_title('Polynomial Basis')
ax1.legend(fontsize=8)
ax1.set_ylim(-4, 4)

# 2. Gaussian basis functions
def gaussian(x, mu, sigma=0.5):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

centers = [-1.5, 0, 1.5]
for i, mu in enumerate(centers):
    ax2.plot(x, gaussian(x, mu), label=r'$\phi_{' + str(i) + r'}(x) = e^{-(x-' + str(mu) + r')^2/2\sigma^2}$')
ax2.set_title('Gaussian RBF')
ax2.legend(fontsize=8)

# 3. Sigmoid basis functions
def sigmoid(x, beta=1, theta=0):
    return 1 / (1 + np.exp(-beta * (x - theta)))

thetas = [-1.5, 0, 1.5]
for i, theta in enumerate(thetas):
    ax3.plot(x, sigmoid(x, theta=theta), label=r'$\phi_{' + str(i) + r'}(x) = \sigma(x-' + str(theta) + r')$')
ax3.set_title('Sigmoid')
ax3.legend(fontsize=8)

# 4. Fourier basis functions
frequencies = [1, 2, 3]
for i, freq in enumerate(frequencies):
    ax4.plot(x, np.sin(freq * np.pi * x), label=r'$\phi_{' + str(2*i) + r'}(x) = \sin(' + str(freq) + r'\pi x)$')
    ax4.plot(x, np.cos(freq * np.pi * x), label=r'$\phi_{' + str(2*i+1) + r'}(x) = \cos(' + str(freq) + r'\pi x)$')
ax4.set_title('Fourier Basis')
ax4.legend(fontsize=8)

# 5. Wavelets (Mexican Hat / Ricker wavelet)
def ricker_wavelet(x, sigma):
    A = 2 / (np.sqrt(3 * sigma) * np.pi**0.25)
    return A * (1 - (x/sigma)**2) * np.exp(-(x**2)/(2*sigma**2))

sigmas = [0.5, 1.0, 1.5]
for i, sigma in enumerate(sigmas):
    ax5.plot(x, ricker_wavelet(x, sigma), label=r'$\sigma=' + str(sigma) + r'$')
ax5.set_title('Wavelet Basis (Mexican Hat)')
ax5.legend(fontsize=8)

# 6. Polynomial Splines
def spline(x, knot, degree=3):
    return np.maximum(0, (x - knot))**degree

knots = [-1, 0, 1]
for i, knot in enumerate(knots):
    ax6.plot(x, spline(x, knot), label=r'$\phi_{' + str(i) + r'}(x) = \max(0, x-' + str(knot) + r')^3$')
ax6.set_title('Cubic Splines')
ax6.legend(fontsize=8)

# 7. Logistic Basis
def logistic_basis(x, center, width=1.0):
    return 1 / (1 + np.exp(-(x - center)/width))

centers = [-1.5, 0, 1.5]
for i, center in enumerate(centers):
    ax7.plot(x, logistic_basis(x, center), label=r'$c=' + str(center) + r'$')
ax7.set_title('Logistic Basis')
ax7.legend(fontsize=8)

# 8. Student's t Basis
def students_t_basis(x, center, df=1.0):
    return 1 / (1 + ((x - center)/df)**2)

centers = [-1.5, 0, 1.5]
for i, center in enumerate(centers):
    ax8.plot(x, students_t_basis(x, center), label=r'$c=' + str(center) + r'$')
ax8.set_title("Student's t Basis")
ax8.legend(fontsize=8)

# 9. Periodic Basis
def periodic_basis(x, freq, phase):
    return 0.5 * (1 + np.cos(freq * x + phase))

frequencies = [1, 2, 3]
phases = [0, np.pi/2, np.pi]
for i, (freq, phase) in enumerate(zip(frequencies, phases)):
    ax9.plot(x, periodic_basis(x, freq, phase), 
             label=r'$f=' + str(freq) + r', \phi=' + f'{phase:.1f}' + r'$')
ax9.set_title('Periodic Basis')
ax9.legend(fontsize=8)

# Adjust layout and save
plt.tight_layout()
plt.savefig('basis_functions_visualization.pdf', bbox_inches='tight', dpi=300)
plt.close() 