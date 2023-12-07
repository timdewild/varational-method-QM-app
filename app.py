import streamlit as st
import plotly.graph_objects as go
from plotly.tools import make_subplots
import numpy as np

## Nice colors https://github.com/plotly/plotly.py/issues/2192 



class TrueWaveFunction:
    def __init__(self):
        self.Psi = None
    
    def get_wavefunction(self,x):
        self.Psi = ( 1/np.pi ) ** 0.25 * np.exp( -x**2/2 )
        return self.Psi

class TrialWaveFunctions:

    options = ['Logistic', 'HyperbolicSecant', 'RaisedCosine']

    plotranges = {
        'Logistic':         {'psi': [-5,5,0,1.25,1,0.2], 'energy': [0.2,1.6,0,2.5,0.2,0.5], 'Nx': 1000, 'Ns': 1000},
        'HyperbolicSecant': {'psi': [-5,5,0,1.25,1,0.2], 'energy': [0.5,2,0,2.5,0.25,0.5],  'Nx': 1000, 'Ns': 1000},
        'RaisedCosine':     {'psi': [-5,5,0,1.25,1,0.2], 'energy': [1,4,0,1.6,0.5,0.4],     'Nx': 1000, 'Ns': 1000}
    }

    def __init__(self, name):
        self.name = name
        self.psi = None
        self.energy = None
        self.plotrange_psi = None
        self.plotrange_E = None
        self.xarray = None
        self.sarray = None

    def sech(self,x):
        return 1/np.cosh(x)

    def get_wavefunction(self,x,s):

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
    
    def get_energy(self,s):

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

def generate_figure(plotrange_psi, plotrange_energy):

    # Initialize figure with subplots
    fig2 = make_subplots(
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
                ),
            row = 1,
            col = 2
            )

    # in figure data, 'skip' the permanent curves (which should be visible by default, and set the first trace of the animated curves to visible)
    fig.data[N_parmanent_curves].visible = True
    fig.data[N_parmanent_curves + N_steps].visible = True

    # Create and add slider
    steps = []
    for i in range(N_steps):
        step = dict(
            method="update",
            args=[              
                {"visible": [True] * N_parmanent_curves + [False] * N_animated_curves*N_steps},
                ],  
            label = str(round(scale_param_list[i],2))
        )
        step["args"][0]["visible"][N_parmanent_curves + i] = True
        step["args"][0]["visible"][N_parmanent_curves + N_steps + i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": '<i>s</i> = '},
        pad={"t": 60},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders, 
    )
    
    return fig


### APP CONTENT ###
st.title('Variational Method and Quantum Harmonic Oscillator')

dist_name = st.selectbox(
    'Choose a trial wavefunction based on the distributions below:',
    TrialWaveFunctions.options)

fig = apply_variational_method(dist_name)

st.plotly_chart(fig)








