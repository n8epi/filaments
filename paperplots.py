# Supporting plots for Filament Plots for Data Visualization
# Author: Nate Strawn, Dept. Math/Stats @ Georgetown University
# Description: Standard matrix of scatterplots, parallel plots, Andrew's plots, and tensor visualization
# License: MIT


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn.datasets as datasets
import sklearn.preprocessing as pp
from sklearn.preprocessing import LabelEncoder

def andrews1D(X, y, cs, labels):
    n_samples = 256
    t = np.linspace(0,1,n_samples)

    N = X.shape[0]
    d = X.shape[1]

    andrews_matrix = np.zeros((d, n_samples))
    andrews_matrix[0,:] = 1
    k = 1
    kappa = 1

    while k < d:
        andrews_matrix[k,:] = np.sqrt(2)*np.cos(2*np.pi*kappa*t)
        k += 1
        if k < d:
            andrews_matrix[k,:] = np.sqrt(2)*np.sin(2*np.pi*kappa*t)
            k += 1
            kappa += 1

    F = X @ andrews_matrix
    transformer = pp.StandardScaler()
    w = transformer.fit_transform(X)
    u, s, v = np.linalg.svd(w,  full_matrices=False)
    W = w @ andrews_matrix
    G = (u * s) @ andrews_matrix

    label_legend = [True] * len(labels)

    fig = make_subplots(rows=1, cols=3)

    for n in range(0,N):
        fig.add_trace(go.Scatter(
            x=t,
            y=F[n,:],
            mode='lines',
            line=dict(color=cs[y[n]], width=3),
            showlegend = label_legend[y[n]],
            legendgroup = labels[y[n]],
            name = labels[y[n]]),
            row = 1,
            col = 1
        )
        label_legend[y[n]] = False

        fig.add_trace(go.Scatter(
            x=t,
            y=W[n,:],
            mode='lines',
            line=dict(color=cs[y[n]], width=3),
            showlegend=label_legend[y[n]],
            legendgroup=labels[y[n]],
            name=labels[y[n]]),
            row=1,
            col=2
        )

        fig.add_trace(go.Scatter(
            x=t,
            y=G[n,:],
            mode='lines',
            line=dict(color=cs[y[n]], width=3),
            showlegend=label_legend[y[n]],
            legendgroup=labels[y[n]],
            name=labels[y[n]]),
            row=1,
            col=3
        )

    fig.update_layout(font_family="Computer Modern",
                      font_size=30
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()

def parallel(X, y, cs, features, labels):

    transformer = pp.StandardScaler()
    w = transformer.fit_transform(X)

    N = X.shape[0]
    d = X.shape[1]
    t = np.array(range(d))

    label_legend = [True] * len(labels)

    fig = make_subplots(rows=1, cols=1)

    for n in range(0,N):
        fig.add_trace(go.Scatter(
            x=t,
            y=w[n,:],
            mode='lines',
            line=dict(color=cs[y[n]], width=3),
            showlegend = label_legend[y[n]],
            legendgroup = labels[y[n]],
            name = labels[y[n]]),
            row = 1,
            col = 1
        )
        label_legend[y[n]] = False

    fig.update_layout(font_family="Computer Modern",
                      font_size=30,
                      xaxis = dict(
                          tickmode = 'array',
                          tickvals = t,
                          ticktext = features
                      )
                      )

    fig.show()

def plotAndrewsTensor2D(d):

    n_samples = 256
    t = np.linspace(0,1,n_samples)

    andrews_tensor = np.zeros((2, d, n_samples))

    k = 1
    while k <= d:
        andrews_tensor[0,k-1,:] = np.sqrt(2)*np.cos(2*np.pi*k*t)
        andrews_tensor[1,k-1,:] = np.sqrt(2)*np.sin(2*np.pi*k*t)
        k += 1

    fig = go.Figure()

    tau = t[150]
    ii = []
    jj = []

    for j in range(2):
        for k in range(d):
            fig.add_trace(go.Scatter3d(
            x=-k*np.ones(n_samples),
            y=t,
            z=(4*j)+andrews_tensor[j-1, d-k-1, :],
            mode='lines',
            opacity=0.5,
            line=dict(color='blue', width=8),
            showlegend=False)
            )
            ii.append(-k)
            jj.append((4*j)+andrews_tensor[j-1, d-k-1, 150])

    u, v = np.mgrid[-2:1,-2:7]

    fig.add_trace(go.Surface(x=u,
                             y=tau*np.ones(u.shape),
                             z=v,
                             opacity=0.25,
                             surfacecolor=np.ones(u.shape)/2,
                             showscale=False
                             ))

    fig.update_layout(font_family="Computer Modern",
                      font_size=20,
                      scene=dict(
                          xaxis=dict(showgrid=False, ticks='', showticklabels=False, title='coordinates slices'),
                          yaxis=dict(showgrid=False, ticks='', showticklabels=False, title='time slices'),
                          zaxis=dict(showgrid=False, ticks='', showticklabels=False, title='components slices'),
                      )
                      )

    fig.add_trace(go.Scatter3d(
        x=ii,
        y=[tau]*2*d,
        z=jj,
        opacity=1,
        mode='markers',
        marker=dict(color='red', size=12),
        showlegend=False)
    )

    fig.show()


### some code now...

iris = datasets.load_iris()
X = iris.data[:, :4]

le = LabelEncoder()
y = le.fit_transform(iris.target)
labels = iris.target_names
print(labels)
cs = ['#a6cee3','#1f78b4','#b2df8a']

andrews1D(X, y, cs, labels)

# The scatterplot matrix
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
print(features)
parallel(X, y, cs, features, labels)
#print(y)

fig = go.Figure(data=go.Splom(
    dimensions=[dict(label='sepal length',
                     values=X[y == 0,0]),
                dict(label='sepal width',
                     values=X[y == 0,1]),
                dict(label='petal length',
                     values=X[y == 0,2]),
                dict(label='petal width',
                     values=X[y == 0,3])],
    showlegend=True,
    showupperhalf=False, # remove plots on diagonal
    marker=dict(color=cs[0],
                showscale=False, # colors encode categorical variables
                line_color='white', line_width=0.5),
    name='setosa'
))

fig.add_trace(go.Splom(
    dimensions=[dict(label='sepal length',
                     values=X[y == 1,0]),
                dict(label='sepal width',
                     values=X[y == 1,1]),
                dict(label='petal length',
                     values=X[y == 1,2]),
                dict(label='petal width',
                     values=X[y == 1,3])],
    showlegend=True,
    showupperhalf=False, # remove plots on diagonal
    marker=dict(color=cs[1],
                showscale=False, # colors encode categorical variables
                line_color='white', line_width=0.5),
    name='versicolor'
))

fig.add_trace(go.Splom(
    dimensions=[dict(label='sepal length',
                     values=X[y == 2,0]),
                dict(label='sepal width',
                     values=X[y == 2,1]),
                dict(label='petal length',
                     values=X[y == 2,2]),
                dict(label='petal width',
                     values=X[y == 2,3])],
    showlegend=True,
    showupperhalf=False, # remove plots on diagonal
    marker=dict(color=cs[2],
                showscale=False, # colors encode categorical variables
                line_color='white', line_width=0.5),
    name='virginica'
))

fig.update_layout(
    title='Matrix of Scatterplots for Iris Dataset',
    font_family="Computer Modern",
    font_size=30,
    width=1200,
    height=1200,
    legend=dict(itemsizing='constant',
                orientation="h",
                yanchor="top",
                y=0.9,
                xanchor="right",
                x=0.9,
                bordercolor="Black",
                borderwidth=4)
)

#fig.show()

# Andrews Phi plot
#plotAndrewsTensor2D(3)

transformer = pp.StandardScaler()
w = transformer.fit_transform(X)
u,s,v = np.linalg.svd(w)
print(v)

# Example of singular values for the iris mappings
n_samples = 256
t = np.linspace(0,1,n_samples)

psi = np.zeros((2,4,n_samples))
for k in range(4):
    psi[0,k,:] = np.cos(2*np.pi*(k+1)*t)/np.sqrt(2) # sqrt(2) / sqrt(d) for d=4
    psi[1,k,:] = np.sin(2*np.pi*(k+1)*t)/np.sqrt(2)

phi = np.zeros((2,4,n_samples))
for k in range(4):
    th = np.pi * (k+1)**2 / 8
    c = np.cos(th)
    s = np.sin(th)
    u = np.array([[c,-s],[s,c]])
    phi[:,k,:] = u @ psi[:,k,:]

psi_sigma = np.zeros((2, n_samples))
phi_sigma = np.zeros((2, n_samples))

for tau in range(n_samples):
    u, s, v = np.linalg.svd(psi[:,:,tau])
    psi_sigma[:, tau] = s
    u, s, v = np.linalg.svd(phi[:,:,tau])
    phi_sigma[:, tau] = s

fig = go.Figure()
fig.add_trace(go.Scatter(
        x=t,
        y=psi_sigma[0,:]-psi_sigma[1,:],
        mode='lines',
        line=dict(color='blue', width=3),
        name=r'$\sigma_1(\Psi(t))-\sigma_2(\Psi(t))$'
))

fig.add_trace(go.Scatter(
    x=t,
    y=phi_sigma[0,:]-phi_sigma[1,:],
    mode='lines',
    line=dict(color='green', width=3),
    name=r'$\sigma_1(\Phi(t))-\sigma_2(\Phi(t))$'
))

fig.update_layout(font_family="Computer Modern",
                  font_size=30,
                  xaxis_title="time slice r$t$",
                  legend=dict(font_size=16,
                              itemsizing='constant',
                              orientation="v",
                              yanchor="top",
                              y=1.0,
                              xanchor="right",
                              x=0.75),
                  )

fig.show()



