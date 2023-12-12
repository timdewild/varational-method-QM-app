import streamlit as st


st.title('Variational Principle in Quantum Mechanics')
st.write(
    """
    The variational principle of quantum mechanics provides a way of approximating the ground state (i.e. lowest energy eigenstate) and its energy, for systems that are not solvable analytically. 
    It is a unique alternative to perturbative approaches, which only apply in case perturbations are small.
    In this applet, we introduce the variational method of quantum mechanics by applying it to the ground state of the simple harmonic oscillator. 
    Taking this system as our playground, which is indeed solvable analytically, we can compare the variational estimates to the exact results. 

    ##### The Variational Theorem
    Given the Hamiltonian operator $H$ governing the studied system and any normalizable function $\Psi$ (the trial wavefunction), we define the functional:
    $$
    E[\Psi] = \\frac{\langle \Psi| H|\Psi\\rangle}{\langle \Psi|\Psi\\rangle}.
    $$
    The variational theorem states that for any trial wavefunction $E[\Psi]\geq E_0$, where $E_0$ is the true ground state energy. 
    The inquality only saturates ($E=E_0$) if our trial wavefunction exactly equals the true ground state wavefunction of the system. 

    ##### In Practice
    How does this work in practice? After all, the theorem does not tell us what trial wavefunction we should pick. 
    The trick is to cook up a trial wavefunction that respects the key features of system's potential. 
    On general grounds, the wavefunction should have largest support where the potential is smallest, and vice versa. 
    For a symmetric potential, the ground state is symmetric as well and has no nodes. 
    If we pick a smart trial function, practice shows that we can get remarkably accurate estimates for the ground state energy. 
    However, make no mistake: the variational method only gives an upper bound on the true ground state energy. 
    Unless someone tells you the true ground state, there is no way of telling how far off you are from the correct value. 

    Despite these limitations, the variational principle can be very useful. 
    Consider a family of trial functions $\Psi(s)$ depending on a scaling parameter $s$. 
    (Nothing restricts from including multiple parameters $s_i$, but for simplicity we resctrict to one here.)
    For example, if you choose a Bell curve as trial function, $s$ could determine the spread of the curve. 
    We can now define:
    $$
    E(s) = \\frac{\langle \Psi(s)| H|\Psi(s)\\rangle}{\langle \Psi(s)|\Psi(s)\\rangle}.
    $$
    Note that we still have, 
    $$
    E(s)\geq E_0,\quad\mathrm{for\;all}\;s
    $$
    on account of the variational theorem. 
    The most stringent bound on the the ground state energy (i.e. the best estimate for $E_0$) comes from the minimum value of $E(s)$. 
    In other words, we *vary* $s$ until we find the find the minimum of $E(s)$ (it's in the name). Mathematically:
    $$
    \\frac{\partial E(s)}{\partial s}\\bigg|_{s=s_\\ast} = 0. 
    $$

    """
)