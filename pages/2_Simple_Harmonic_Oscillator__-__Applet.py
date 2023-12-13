import streamlit as st
import plotly.graph_objects as go
import plotly.tools as to
import numpy as np
import math as ma

## Nice colors https://github.com/plotly/plotly.py/issues/2192 
## Unicode for math symbols https://www.compart.com/en/unicode/html 



class TrueWaveFunction:
    def __init__(self):
        self.Psi = None
    
    def get_wavefunction(self,x):
        self.Psi = ( 1/np.pi ) ** 0.25 * np.exp( -x**2/2 )
        return self.Psi

class TrialWaveFunctions:

    options = ['Gaussian', 'Logistic', 'HyperbolicSecant', 'RaisedCosine']

    plotranges = {
        'Gaussian':         {'psi': [-5,5,0,1.25,1,0.2], 'energy': [0.5,    3,      0,      0.8,    0.5,    0.2],   'Nx': 1000, 'Ns': 1000},
        'Logistic':         {'psi': [-5,5,0,1.25,1,0.2], 'energy': [0.2,    1.6,    0,      2.5,    0.2,    0.5],   'Nx': 1000, 'Ns': 1000},
        'HyperbolicSecant': {'psi': [-5,5,0,1.25,1,0.2], 'energy': [0.5,    2,      0,      2.5,    0.25,   0.5],   'Nx': 1000, 'Ns': 1000},
        'RaisedCosine':     {'psi': [-5,5,0,1.25,1,0.2], 'energy': [1,      4,      0,      1.6,    0.5,    0.4],   'Nx': 1000, 'Ns': 1000}
    }

    def __init__(self, name):
        self.name = name
        self.psi = None
        self.overlap = None
        self.energy = None
        self.plotrange_psi = None
        self.plotrange_E = None
        self.xarray = None
        self.sarray = None

    def sech(self,x):
        return 1/np.cosh(x)
    
    def get_name(self):
        return self.name

    def get_wavefunction(self,x,s):

        if self.name == 'Gaussian':
            N = (s/np.pi) ** 0.25
            self.Psi = N * np.exp( -0.5 * s * x**2 )

        if self.name == 'Logistic':
            N = np.sqrt(6*s)
            self.Psi = N / (4*s) * self.sech(x / 2 / s) ** 2

        if self.name == 'HyperbolicSecant':
            N = np.sqrt(np.pi * s)
            self.Psi = N / (2*s) * self.sech(np.pi / 2 * x / s)

        if self.name == 'RaisedCosine':
            N = np.sqrt(4*s / 3)
            self.Psi = np.where(np.abs(x) <= s, N / (2*s) * (1 + np.cos(np.pi * x / s)), 0)

        return self.Psi
    
    def get_formulas(self):
        if self.name == 'Gaussian':
            wf_formula = r"""
            $$
            \small
            \begin{align*}
            \Psi(x,s) &= N(s)\;e^{-\frac{1}{2}sx^2},\\
            N(s) &= (s/\pi)^{1/4}.
            \end{align*}
            $$
            """
            Es_formula = r"""
            $$
            \small
            \begin{align*}
            E(s) = \frac{1}{4}\Big(\frac{1}{s}+s\Big).
            \end{align*}
            $$
            """
            s_E_star = r"""
            $$
            \small
            \begin{align*}
            s_\ast = 1,\quad\quad E_\ast = \frac{1}{2}.
            \end{align*}
            $$
            """

        if self.name == 'Logistic':
            wf_formula = r"""
            $$
            \small
            \begin{align*}
            \Psi(x,s) &= \frac{N(s)}{4s}\mathrm{sech}^2(x/2s),\\
            N(s) &= \sqrt{6s}.
            \end{align*}
            $$
            """
            Es_formula = r"""
            $$
            \small
            \begin{align*}
            E(s) = \frac{1}{10s^2}+\frac{3s^2}{16}(\pi^2-6).
            \end{align*}
            $$
            """
            s_E_star = r"""
            $$
            \small
            \begin{align*}
            s_\ast = \bigg[\frac{8}{15(\pi^2-6)}\bigg]^{1/4},\quad\quad E_\ast = 0.539.
            \end{align*}
            $$
            """

        if self.name == 'HyperbolicSecant':
            wf_formula = r"""
            $$
            \small
            \begin{align*}
            \Psi(x,s) &= \frac{N(s)}{2s}\mathrm{sech}(\pi x/2s),\\
            N(s) &= \sqrt{\pi s}.
            \end{align*}
            $$
            """
            Es_formula = r"""
            $$
            \small
            \begin{align*}
            E(s) = \frac{\pi^2}{24s^2}+\frac{s^2}{6},
            \end{align*}
            $$
            """
            s_E_star = r"""
            $$
            \small
            \begin{align*}
            s_\ast = (\pi^2/4)^{1/4},\quad\quad E_\ast = 0.524.
            \end{align*}
            $$
            """


        if self.name == 'RaisedCosine':
            wf_formula = r"""
            $$
            \small
            \begin{align*}
            \Psi(x,s) &= \frac{N(s)}{2s}\big[1+\cos(\pi x/s)\big],\quad \,x\in[-s,s]\\
            &= 0,\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad x\;\notin[-s,s]\\
            N(s) &= \sqrt{4s/3}
            \end{align*}
            $$
            """
            Es_formula = r"""
            $$
            \small
            \begin{align*}
            E(s) = \frac{\pi^2}{6s^2}+\frac{s^2}{6}\bigg(1-\frac{15}{2\pi^2}\bigg).
            \end{align*}
            $$
            """
            s_E_star = r"""
            $$
            \small
            \begin{align*}
            s_\ast = \bigg[\frac{\pi^2}{1-15/2\pi^2}\bigg]^{1/4},\quad\quad E_\ast = 0.513.
            \end{align*}
            $$
            """


        return wf_formula, Es_formula, s_E_star

    
    def get_overlap(self,s):
        """Gives inner-product <trial|true>**2 of trial function with true (Gaussian) function. Fitting functions from mathematica."""

        if self.name == 'Gaussian':
            overlap = np.sqrt(1/np.pi) * s ** (1/4) * np.sqrt(2*np.pi / (s + 1))

        if self.name == 'Logistic':
            overlap = -0.0933275 + 4.74778 * s - 7.25974 * s **2 + 4.64651 * s **3 - 1.13741 * s **4 + 0.000456442 * (s **5 + s **6 + s **7 + s **8 + s **9 + s **10 + s **11)

        if self.name == 'HyperbolicSecant':
            overlap = -0.109648 + 2.54709 * s - 2.14077 *s**2 + 0.795556 *s **3 - 0.118317 *s**4 + 4.43549e-6 * (s**5 + s**6 + s**7 + s**8 + s**9 + s**10 + s**11)

        if self.name == 'RaisedCosine':
            overlap = -0.139298 + 1.11729 * s - 0.34744 *s**2 + 0.0317781 *s**3 + 0.000475911 *s**4 - 6.85187e-9 * (s**5 + s**6 + s**7 + s**8 + s**9 + s**10 + s**11)
        
        if self.name != 'Gaussian':
            overlap[overlap>1] = 0.9999

        self.overlap = overlap

        return self.overlap 

    
    def get_energy(self,s):
        if self.name == 'Gaussian':
            self.energy = 1 / 4 *(1/s + s)

        if self.name == 'Logistic':
            self.energy = 1 / 10 / s ** 2 + 3 / 16 * (np.pi ** 2 - 6) * s ** 2

        if self.name == 'HyperbolicSecant':
            self.energy = np.pi**2 / 24 / s**2 + s**2 / 6

        if self.name == 'RaisedCosine':
            self.energy = np.pi**2 / 6 / s**2 + s**2 / 6 * (1 - 15 / 2 / np.pi**2)

        return self.energy
    
    def get_xarray(self):
        xmin = TrialWaveFunctions.plotranges[self.name]['psi'][0] 
        xmax = TrialWaveFunctions.plotranges[self.name]['psi'][1] 
        Nx = TrialWaveFunctions.plotranges[self.name]['Nx']

        self.xarray = np.linspace(xmin, xmax, Nx)

        return self.xarray
    
    def get_sarray(self):
        """Numpy array with s values for plotting only."""

        smin = TrialWaveFunctions.plotranges[self.name]['energy'][0] 
        smax = TrialWaveFunctions.plotranges[self.name]['energy'][1] 
        Ns = TrialWaveFunctions.plotranges[self.name]['Ns']

        self.sarray = np.linspace(smin, smax, Ns)

        return self.sarray

    def get_scale_array(self, ds=0.02):
        """Numpy array with s values for slider only."""

        smin = TrialWaveFunctions.plotranges[self.name]['energy'][0]
        smax = TrialWaveFunctions.plotranges[self.name]['energy'][1]

        N = int((smax-smin)/ds)
        self.sarray = np.linspace(smin,smax, N+1)

        return self.sarray

    def get_plotrange_psi(self):
        return TrialWaveFunctions.plotranges[self.name]['psi']
    
    def get_plotrange_energy(self):
        return TrialWaveFunctions.plotranges[self.name]['energy']
    

##################



def truncate(x,d):
    trunc_factor = 10**d
    x_trunc = ma.trunc(x * trunc_factor) / trunc_factor
    return x_trunc

def plot_title(overlap_value, energy_value):
    tit = "<b>Overlap &#124;&#10216;&#936;&#124;&#936;<sub>0</sub>&#10217;&#124;<sup>2</sup> = {:.3f} and Energy <i>E</i> = {:.3f} </b>".format(overlap_value, energy_value)
    return tit




def generate_figure(plotrange_psi, plotrange_energy):

    # Initialize figure with subplots
    fig2 = to.make_subplots(
        rows=1, 
        cols=2, 
        horizontal_spacing=0.15
    )

    x1min, x1max, y1min, y1max, dx1, dy1 = plotrange_psi #[-5,5,0,1.25,1,0.2]
    x2min, x2max, y2min, y2max, dx2, dy2 = plotrange_energy #[0.5,2,0,2.5,0.25,0.5]

    # Update xaxis properties
    fig2.update_xaxes(
        row=1, 
        col=1, 
        range=[x1min,x1max], 
        showline=True, 
        linecolor='black', 
        mirror=True, 
        constrain='domain',
        ticks='inside',
        ticklen=5,
        tickwidth=1,
        tickcolor='black',
        tick0=x1min,
        dtick=dx1,
        showgrid = True, 
        gridwidth=1.5, 
        gridcolor='white',
        title_text="<i>x</i>"
    )

    fig2.update_xaxes(
        row=1, 
        col=2, 
        range=[x2min, x2max], 
        showline=True, 
        linecolor='black', 
        mirror=True, 
        constrain='domain',
        ticks='inside',
        ticklen=5,
        tickwidth=1,
        tickcolor='black',
        tick0=x2min,
        dtick=dx2,
        showgrid = True, 
        gridwidth=1.5, 
        gridcolor='white',
        title_text="<i>s</i>"
    )

    # Update yaxis properties
    fig2.update_yaxes(
        row=1, col=1, 
        range=[y1min,y1max], 
        showline=True, 
        linecolor='black', 
        mirror=True, 
        constrain = 'domain', 
        ticks='inside',
        ticklen=5,
        tickwidth = 1,
        tickcolor='black',
        automargin=True,
        tick0=y1min,
        dtick=dy1,  
        showgrid = True,
        gridwidth=1.5, 
        gridcolor='white',
        title_text="Wavefunction &#936;(<i>x,s</i>)",   
    )

    fig2.update_yaxes(
        row=1, 
        col=2, 
        range=[y2min, y2max], 
        showline=True, 
        linecolor='black', 
        mirror=True, 
        constrain='domain', 
        ticks='inside',
        ticklen=5,
        tickwidth = 1,
        tickcolor='black',
        automargin=True,
        tick0=y2min,
        dtick=dy2,
        showgrid = True,
        gridwidth=1.5, 
        gridcolor='white',
        title_text="Energy <i>E(s)</i> [&#8463;&#969;]"
    )

    # Update layout
    fig2.update_layout( 
        showlegend=False, 
        margin = dict(l = 12, r = 12), 
        plot_bgcolor = "#e3ecf7",

    )

    return fig2

##################

def apply_variational_method(trial_name):

    trial_wavefunction = TrialWaveFunctions(trial_name)
    true_wavefunction = TrueWaveFunction()

    x_list = trial_wavefunction.get_xarray()
    s_list = trial_wavefunction.get_sarray()

    ### GENERATE GRAPHING OBJECT
    fig = generate_figure(trial_wavefunction.get_plotrange_psi(), trial_wavefunction.get_plotrange_energy())

    # number of permanent traces
    N_parmanent_curves = 2

    # number of values grid of scale_param (s)  N_steps
    scale_param_list = trial_wavefunction.get_scale_array()
    N_steps = len(scale_param_list)

    # number of animated curves in the WHOLE figure (so ALL subplots combined): number of traces that should be toggled on/off is then N_animated_curves * N_steps
    N_animated_curves = 2 

    #overlap data for s values
    overlap = trial_wavefunction.get_overlap(scale_param_list)


    ### Define traces for permanent curves ### 

    # background gaussian
    fig.add_trace(
        go.Scatter(
            visible = True,
            x = x_list,
            y = true_wavefunction.get_wavefunction(x_list),
            mode = 'markers',
            marker = dict(color='MidnightBlue', size = 1.5)
        ),
        row = 1,
        col = 1
    )

    # energy curve
    fig.add_trace(
        go.Scatter(
            visible = True,
            x = s_list,
            y = trial_wavefunction.get_energy(s_list),
            mode = 'markers',
            marker = dict(color='IndianRed', size = 1.5)
        ),
        row = 1,
        col = 2
    )

    ### Generate traces for animated curves ### 

    for scale_param in scale_param_list:
        fig.add_trace(
            go.Scatter(
                visible = False,
                x = x_list,
                y = trial_wavefunction.get_wavefunction(x_list, scale_param),
                mode = 'lines',
                line = dict(color='Crimson', width = 1.5)
                ),
            row = 1,
            col = 1
            )
        
    for scale_param in scale_param_list:
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=[scale_param],
                y=[trial_wavefunction.get_energy(scale_param)],
                mode='markers',
                marker=dict(
                    size=6,
                    color='LightSalmon',
                    line=dict(
                        width=1.5,
                        color='Crimson'
                        )
                    )
                ),
            row = 1,
            col = 2
            )

    # in figure data, 'skip' the permanent curves (which should be visible by default, and set the first trace of the animated curves to visible)
    fig.data[N_parmanent_curves].visible = True
    fig.data[N_parmanent_curves + N_steps].visible = True

    # Create and add slider
    initial_step_index = 0

    # Define initial overlap value trunctated at 3 decimal places
    overap_value_0 = truncate(overlap[initial_step_index],3)
    energy_value_0 = trial_wavefunction.get_energy(scale_param_list[initial_step_index])



    steps = []
    for i, s in enumerate(scale_param_list):
        overlap_value = truncate(overlap[i],3)
        energy_value = trial_wavefunction.get_energy(s)
        step = dict(
            method="update",
            args=[              
                {"visible": [True] * N_parmanent_curves + [False] * N_animated_curves * N_steps},
                {"title.text": plot_title(overlap_value, energy_value)} 
                ],  
            label = str(round(s,2))
        )
        step["args"][0]["visible"][N_parmanent_curves + i] = True
        step["args"][0]["visible"][N_parmanent_curves + N_steps + i] = True
        steps.append(step)

    sliders = [dict(
        active=initial_step_index,
        currentvalue={"prefix": '<i>s</i> = ', "font": {"size": 14}},
        pad={"t": 60},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders, 
        title={
            'text': plot_title(overap_value_0,energy_value_0), 
            'x': 0.5, 
            'xanchor': 'center',
            'yanchor': 'top'
            }
    )
    
    return fig

### APP CONTENT ###


with st.sidebar:
    st.write(
        """
        ## Select Trial Wavefunction
        Choose trial wavefunction from options below:
        """
        )
    dist_name = st.selectbox(
        '',
        TrialWaveFunctions.options, 
        label_visibility='collapsed'
        )
    
    trial_wavefunction = TrialWaveFunctions(dist_name)
    wf_formula, Es_formula, s_E_star = trial_wavefunction.get_formulas()

    st.write(
        f"""
        ## Selected: {trial_wavefunction.get_name()}
        The selected trial wavefunction $\Psi$ with normalization factor $N$ is:
        """
    )
    st.markdown(
        wf_formula
    )
    st.write(
        """The energy as function of the scale parameter $s$ is:
        """
    )
    st.markdown(
        Es_formula
    )
    st.write(
        """The energy is minimized at $s=s_\\ast$ with value $E_\\ast$:"""
    )
    st.markdown(
        s_E_star
    )

st.write(
    f"""
    #### Variational Method SHO: {trial_wavefunction.get_name()} Trial Wavefunction
    """
)

fig = apply_variational_method(trial_wavefunction.get_name())

st.plotly_chart(fig)

st.write(
    """
    ###### Description Applet
    The applet above shows two panels and a slider for the scale parameter $s$ to vary the shape of the trial function. 
    By default the Gaussian trial function is selected, to select a different trial function, open the sidebar on your left and choose from the four options. 
    Details on the selected trial function are shown in the sidebar as well. 
    The left panel shows the trial wavefunction $\Psi$ (dark red) and the true Gaussian ground state wavefunction $\Psi_0$ (dark blue) as function of position $x$. 
    The right panel shows the energy estimate $E(s)$ as function of the scale parameter $s$. 
    By varying the slider for $s$, you will see that the trial function changes shape and the red-orange dot gives the energy for the current $s$-value. 
    The exact value for $E(s)$ is given in the plot title, as well as the overlap with the exact ground state for the simple harmonic oscillator (in natural units):
    $$
    \Psi_0(x) = \\frac{e^{-\\frac{1}{2}x^2}}{\pi^{1/4}}. 
    $$
    Note that the energy becomes minimized when the trial function best resembles the true wavefunction: this is the core of the variational principle. 
    
    """
)








