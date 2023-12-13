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
    The inequality only saturates ($E=E_0$) if our trial wavefunction exactly equals the true ground state wavefunction of the system. 

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
    Note that this only gives most stringent energy $E_\\ast\equiv E(s_\\ast)$ given the chosen family of trial functions $\\Psi(s)$. 
    Choosing a different family $\\tilde{\\Psi}(s)$ may well give a more stringent upper bound $\\tilde{E}_\\ast$.
    This is the essence of the variational game: if you find a trial function with a lower value $E_\\ast$, you won. 
    At least at that moment, until someone finds one with yet a lower energy. 

    ##### How accurate is the Variational Method?
    Above, I mentioned that practice shows you can get remarkably accurate results for the ground state energy. 
    Can we make this statement more formal? 
    Suppose the trial function that minimizes the energy, $\Psi(s_\\ast)$, differs from the true ground state $\Psi_0$ by:
    $$
    \Psi(s_\\ast) = \\frac{1}{\\sqrt{1+\\epsilon^2}}(\Psi_0+\\epsilon\Phi),
    $$
    where $\Phi$ is a normalized state that is orthogonal to the ground state, $\\langle\Phi|\Psi_0\\rangle=0$, and $\epsilon$ is small.
    The approximated energy is then:
    $$
    E_\\ast = \\frac{1}{1+\epsilon^2}\\big[\langle\Psi_0|H|\Psi_0\\rangle+\epsilon (\langle\Psi_0|H|\Phi\\rangle + \langle\Phi|H|\Psi_0\\rangle) + \epsilon^2\langle \Phi|H|\Phi\\rangle   \\big].
    $$
    The key observation is that terms linear in $\epsilon$ vanish because $\\langle \Phi|H|\Psi_0\\rangle = E_0\\langle \Phi|\Psi_0\\rangle = 0$. 
    Expanding the normalization factor, we get:
    $$
    E_\\ast = E_0 + \epsilon^2 (\\langle \Phi|H|\Phi\\rangle-E_0)+\\mathcal{O}(\epsilon^4). 
    $$
    In other words, if the difference between the trial wavefunction and the true ground state wavefunction is $\mathcal{O}(\epsilon)$, the difference from the ground state energy is $\mathcal{O}(\epsilon^2)$. 
    This is why the variational method often gives accurate results. However, there is still no way of telling how good our trial wavefunction is. Put differently, there is no way of estimating the size of $\epsilon$. 

    ##### What if the Ground State is known? 
    What if the ground state of the system is actually known and we merely use the system as a playground to demonstrate the variational principle? 
    In that case, a measure for how well the trial function resembles the true wavefunction is the overlap:
    $$
    O(s)\equiv |\\langle \Psi(s)|\Psi_0\\rangle|^2. 
    $$
    Assuming both the trial and true wavefunction are normalized, the closer the overlap gets to unity, the better the trial function resembles the true wavefunction. 
    Beware, only if the trial function $\Psi(s)$ exactly coincides with $\Psi_0$ will the overlap *equal* unity. 

    """
)