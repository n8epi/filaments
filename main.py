# Examples for Filament Plots for Data Visualization
# Author: Nate Strawn, Dept. Math/Stats @ Georgetown University
# Description: We provide 2D Andrew's plots and 3D filament plots for several datasets
# License: MIT

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.datasets as datasets
import sklearn.preprocessing as pp
from sklearn.preprocessing import LabelEncoder

def pathsplit(a):
    '''
    Split a matrix of path data for ingestion into a data frame
    :param a: the matrix of path data
    :return: the path data in tabular format with indexing
    '''
    m,n,r = a.shape
    return np.column_stack((np.repeat(np.arange(m),r), np.tile(np.arange(r),m),(np.reshape(np.stack(a,2),(2,m*r),order='F')).T))

def so3exp(k1,k2):
    '''
    Rodrigues formula for updating based on Frenet-Serret equations
    :param k1: vector of symmetric curvature values for the first component
    :param k2: vector of symmetric curvature values for the second component
    :return: the result from exponentiation of a skew-symmetric matrix
    '''

    # Rodrigues is I + sin(k)A + (1-cos(k))A^2
    # Here, A is skew symmetric with upper diagonal -k2, k1
    k = np.sqrt(k1**2 + k2**2)
    a = k1/k
    b = k2/k
    A = np.array([[0,-a,0],[a,0,b],[0,-b,0]])
    return np.eye(3) + np.sin(k)*A + (1-np.cos(k))*(A@A)

def camera_coordinates(th=0.15, zed=1.0, zoom=0.6):
    '''
    Function for getting scene camera coordinates
    :param th: angle in the x-y plane
    :parma z: lift above z-axis
    :param zoom: zoom indicator
    :return: dictionary of camera information
    '''

    cc = np.cos(2*np.pi*th)
    sc = np.sin(2*np.pi*th)
    xc = zoom*(-1*cc-2*sc)
    yc = zoom*(-2*cc+sc)
    zc = zoom*zed
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xc, y=yc, z=zc)
    )
    return camera

def psi_fcn(d, q):
    '''
    Closures for continuous curve functions
    :param d: dimension of the data
    :param q: curve coefficients vector
    :return: continuous 2D Andrew's plot function (phase shifted)
    '''
    g = np.array(range(d))+1
    qps = g**2/(4*d)
    def psi(s):

        return np.array([np.cos(2*np.pi*(g*s+qps)), np.sin(2*np.pi*(g*s+qps))]) @ q
    return psi

def filamentPlot(X, label_index, labels, color_scale, res = 64, center_index = False, curve_factor=1.0):
    '''
    Filament plot for categorical data
    :param X: data matrix with examples in rows (columns are features)
    :param label_index: integer classes for the categories
    :param labels: category labels
    :param color_scale: colors for the different categories
    :param res: the resolution of the resulting embedding (default is 64 time samples)
    :param center_index: optional index of a data point for ``relativization" of the plot
    :param curve_factor: optional scalar for exaggerating curvatures
    :return: none
    '''

    U,S,V = np.linalg.svd(X, full_matrices=False)
    Y = U @ np.diag(np.sqrt(S))
    if center_index is not False:
        Y = Y - Y[center_index,:]

    # Compute the embedding
    t = np.linspace(0,1,res)
    N = X.shape[0]
    d = X.shape[1]
    Psi = np.zeros((2, d, res))
    for j in range(d):
        k = j+1
        ck = np.cos(2*np.pi*(k**2)/(4*d))
        sk = np.sin(2*np.pi*(k**2)/(4*d))
        U = np.reshape([ck,-sk,sk,ck], (2,2))
        Psi[0, j, :] = np.cos(2*np.pi*k*t)
        Psi[1, j, :] = np.sin(2*np.pi*k*t)
        Psi[:, j, :] = U @ Psi[:,j,:]

    # The final embedding
    E = curve_factor * np.dot(Y,Psi)

    # E is an N by 2 by res matrix; we form the new path matrix F which is N by 3 by res, and update

    dt = 1/(res-1)
    F = np.zeros((N,3,res))
    for n in range(N):
        u_n = np.eye(3)
        psi_n = psi_fcn(d,Y[n, :])
        for r in range(res):
            # Lie-Euler first-order method
            # u_n = so3exp(dt*E[n,0,r], dt*E[n,1,r]) @ u_n

            # Crouch-Grossman third-order method
            t1 = t[r]
            t2 = t1 + (3*dt)/4
            t3 = t1 + (17*dt)/24

            v1 = (24*psi_n(t1)*dt)/17
            v2 = (-2*psi_n(t2)*dt)/3
            v3 = (13*psi_n(t3)*dt)/51

            q1 = so3exp(v1[0], v1[1])
            q2 = so3exp(v2[0], v2[1])
            q3 = so3exp(v3[0], v3[1])

            u_n = q3 @ q2 @ q1 @ u_n

            F[n, :, r] = u_n[1, :]

        F[n, :, :] = np.cumsum(F[n, :, :], axis=1)

    ii = np.argsort(label_index)
    label_index_ = label_index[ii]
    F = F[ii, :, :]
    label_legend = [True] * len(labels)

    fig = go.Figure()

    for n in range(N):
        fig.add_trace(go.Scatter3d(
            x=F[n, 0, :],
            y=F[n, 1, :],
            z=F[n, 2, :],
            mode='lines',
            showlegend=label_legend[label_index_[n]],
            legendgroup=labels[label_index_[n]],
            name=labels[label_index_[n]],
            opacity=0.5,
            line=dict(
                color=color_scale[label_index_[n]],
                width=3
            )
        ))
        label_legend[label_index_[n]] = False

    fig.update_layout(font_family="Computer Modern",
                      font_size=20,
                      legend=dict(itemsizing='constant',
                                  orientation="h",
                                  yanchor="bottom",
                                  y=0.1,
                                  xanchor="center",
                                  x=0.5,
                                  bordercolor="Black",
                                  borderwidth=4),
                      scene_camera=camera_coordinates(th=0.4, zed=-1.0, zoom=0.8))
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    fig.show(renderer="browser")


def andrews2D(X, label_index, labels, color_scale, res = 64, extent=10.0, center_index = False):
    '''
    2D Andrews plots: animated scatter plots, static plots of the curve images, and 2D+1D graph plots
    :param X: data matrix with examples in rows (columns are features)
    :param label_index: integer classes for the categories
    :param labels: category labels
    :param color_scale: colors for the different categories
    :param res: the resolution of the resulting embedding (default is 64 time samples)
    :param extent: symmetric limit for the x and y range of the plot
    :param center_index: optional index of a data point for ``relativization" of the plot
    :return: none
    '''

    # Perform pca on the data matrix
    U,S,V = np.linalg.svd(X, full_matrices=False)
    Y = U @ np.diag(np.sqrt(S))

    # Compute the embedding
    t = np.linspace(0,1,res)
    N = X.shape[0]
    d = X.shape[1]
    Psi = np.zeros((2,d,res))
    for j in range(d):
        k = j+1
        ck = np.cos(2*np.pi*(k**2)/(4*d))
        sk = np.sin(2*np.pi*(k**2)/(4*d))
        U = np.reshape([ck,-sk,sk,ck], (2,2))
        Psi[0,j,:] = np.sin(2*np.pi*k*t)/k
        Psi[1,j,:] = -np.cos(2*np.pi*k*t)/k
        Psi[:,j,:] = U @ Psi[:,j,:]

    # The final embedding
    E = np.dot(Y, Psi)

    if center_index is not False:
        E = E - E[center_index,:,:]

    ii = np.argsort(label_index)
    label_index_ = label_index[ii]
    E = E[ii, :, :]

    # Construct a data frame

    df = pd.DataFrame(pathsplit(E))
    df.columns = ['data_index', 'Time Slice', 'x', 'y']
    df['Class Label'] = [labels[j] for j in np.repeat(label_index_, res).tolist()]

    e = extent
    col_dict = dict(zip(labels, color_scale))

    fig = px.scatter(df, x="x", y="y", animation_frame="Time Slice", animation_group="data_index",
                     color="Class Label", hover_name="data_index", range_x=[-e,e], range_y=[-e,e],
                     color_discrete_map=col_dict
                     )
    fig.update_layout(font_family="Computer Modern",
                      font_size=30,
                      legend=dict(itemsizing='constant',
                                  orientation="h",
                                  title=dict(text=''),
                                  bgcolor = 'white',
                                  yanchor="bottom",
                                  y=0.1,
                                  xanchor="center",
                                  x=0.5,
                                  bordercolor="Black",
                                  borderwidth=4),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    fig.show(renderer="browser")


    fig = go.Figure()
    label_legend = [True] * len(labels)

    for n in range(N):
        fig.add_trace(go.Scatter(
            x=df.loc[df["data_index"] == n]["x"],
            y=df.loc[df["data_index"] == n]["y"],
            showlegend=label_legend[label_index_[n]],
            legendgroup=labels[label_index_[n]],
            name=labels[label_index_[n]],
            mode='lines',
            opacity=0.5,
            line=dict(
                color=color_scale[label_index_[n]],
                width=3
            )
        ))
        label_legend[label_index_[n]] = False

    fig.update_layout(font_family="Computer Modern",
                      font_size=30,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(itemsizing='constant',
                                  orientation="h",
                                  title=dict(text=''),
                                  bgcolor = 'white',
                                  yanchor="bottom",
                                  y=0.1,
                                  xanchor="center",
                                  x=0.5,
                                  bordercolor="Black",
                                  borderwidth=4)
                      )
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show(renderer="browser")

    label_legend = [True] * len(labels)

    fig = go.Figure()

    for n in range(N):
        fig.add_trace(go.Scatter3d(
            x=df.loc[df["data_index"] == n]["x"],
            y=df.loc[df["data_index"] == n]["Time Slice"],
            z=df.loc[df["data_index"] == n]["y"],
            showlegend=label_legend[label_index_[n]],
            legendgroup=labels[label_index_[n]],
            name=labels[label_index_[n]],
            mode='lines',
            opacity=0.5,
            line=dict(
                color=color_scale[label_index_[n]],
                width=4
            )
        ))
        label_legend[label_index_[n]] = False


    fig.update_layout(font_family="Computer Modern",
                      font_size=30,
                      legend=dict(itemsizing='constant',
                                  orientation="h",
                                  yanchor="bottom",
                                  y=0.1,
                                  xanchor="center",
                                  x=0.5,
                                  bordercolor="Black",
                                  borderwidth=4),
                      scene_camera=camera_coordinates()
                      )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    fig.show(renderer="browser")

def iris_plots():
    '''
    Iris dataset plots
    :return: none
    '''
    iris = datasets.load_iris()
    X = iris.data[:, :4]
    transformer = pp.StandardScaler()
    X = transformer.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(iris.target)
    labels = iris.target_names
    cs = ['#a6cee3', '#1f78b4', '#b2df8a']
    andrews2D(X, y, labels, cs, extent=1.0)
    filamentPlot(X, y, labels, cs)

def boston_plots():
    '''
    Boston housing dataset plots
    :return: none
    '''
    boston = datasets.load_boston()
    X = boston.data[:, :13]
    transformer = pp.StandardScaler()
    X = transformer.fit_transform(X)
    y = pd.qcut(boston.target, 10, labels=False)
    labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    cs = ['#a50026','#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
    andrews2D(X, y, labels, cs, res=128, extent=1.0)
    filamentPlot(X, y, labels, cs, res=128)

def breast_cancer_plots():
    '''
    Wisconsin breast cancer dataset plots
    :return:
    '''
    bc = datasets.load_breast_cancer()
    X = bc.data[:, :30]
    transformer = pp.StandardScaler()
    X = transformer.fit_transform(X)
    y = bc.target
    cs = ['#a50026', '#313695']
    labels = ['malignant', 'benign']
    andrews2D(X, y, labels, cs, res=128, extent=3.0)
    filamentPlot(X, y, labels, cs, res=128)

def digit_plots():
    '''
    Handwritten digits plots
    :return:
    '''
    dig = datasets.load_digits()
    X = dig.data[:, :64]
    transformer = pp.StandardScaler()
    X = transformer.fit_transform(X)
    y = dig.target
    labels = [str(j) for j in dig.target_names]
    cs = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    andrews2D(X, y, labels, cs, res=256, extent=1.0)
    filamentPlot(X, y, labels, cs, res=256)


if __name__ == '__main__':

    print('Plotting the Iris dataset...')
    iris_plots()

    print('Plotting the Boston dataset...')
    boston_plots()

    print('Plotting the Breast Cancer dataset...')
    breast_cancer_plots()

    print('Plotting the Digits dataset...')
    digit_plots()

