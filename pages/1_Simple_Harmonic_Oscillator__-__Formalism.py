import streamlit as st

st.title('Simple Harmonic Oscillator - Formalism')
st.write(
    """
    We now wish to apply the variational method to the ground state of the simple harmonic oscillator. 
    We will work in units that are natural to the simple harmonic oscillator, that is we set:
    $$
    m=\omega=\hbar\equiv 1,
    $$
    and energy and distances are measured in terms of $\hbar\omega$ and $\sqrt{\hbar/m\omega}$, respectively. 
    The Hamiltonian operator $H$ is then given by:
    $$
    H = -\\frac{1}{2}\\frac{d^2}{dx^2}+\\frac{1}{2}x^2. 
    $$
    For this system, the ground state and energy are well known and given (in natural units) by:
    $$
    \Psi_0(x) = \\frac{e^{-\\frac{1}{2}x^2}}{\pi^{1/4}},\quad\quad\quad E_0 = \\frac{1}{2}. 
    $$


    For convenience, we assume the trial wavefunction $\Psi(s)$ is normalized and depends on only one scale parameter $s$. 
    If it is not normalized, it can be normalized by computing $N^2(s)\equiv 1/\langle\Psi(s)|\Psi(s)\\rangle$ and rescaling $\Psi(s)\\to N(s)\Psi(s)$.
    To apply the variational method, we have to construct:
    $$
    E(s) = \\langle \Psi(s)|H|\Psi(s)\\rangle,
    $$
    and minimize with respect to $s$ to find the most stringest bound on the energy $E_\\ast$ and approximation to the wavefunction $\Psi(s_\\ast)$.

    Inserting the Hamiltonian for the simple harmonic oscillator, we find:
    $$
    \\begin{align*}
        E(s) & = -\\frac{1}{2}\int_{-\infty}^{+\infty} dx\;\Psi^\\ast(x,s) \\frac{d^2\Psi(x,s)}{dx^2} + \\frac{1}{2}\int_{-\infty}^{+\infty} dx\;\Psi^\\ast(x,s) x^2\Psi(x,s) \\\\
        & = +\\frac{1}{2}\int_{-\infty}^{+\infty} dx\;\\frac{d\Psi^\\ast(x,s)}{dx}\\frac{d\Psi(x,s)}{dx} + \\frac{1}{2}\int_{-\infty}^{+\infty} dx\;\Psi^\\ast(x,s) x^2\Psi(x,s).
    \\end{align*}
    $$
    In going from the first to the second line, we integrated the first term by parts. 
    The boundary term vanishes under the assumption that: 
    $$
    \lim_{x\\to\pm\infty}\Psi(x,s) = 0.
    $$
    The advantage of the second line is purely computational: this way we only have to find the first derivative of the trial function.
    In case the trial function is real, the expression further simplifies as $\Psi=\Psi^\\ast$. 


    """
)
