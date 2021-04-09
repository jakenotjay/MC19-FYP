# reads some generated pickle files and generates graphs from them
# LEGACY - probably not too helpful
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def make2dPlots(dataframe):
    fig = make_subplots(rows=2, cols=2)

    fig.add_traces(
        go.Histogram(
            x=dataframe['FibreFraction'],
            name='Fibre Fraction'
            ),
        rows=1, cols=1
    )

    fig.add_traces(
        go.Histogram(
            x=dataframe['MeanFibreArea'],
            name='Mean Fibre Area'),
        rows=1, cols=2
    )

    fig.add_traces(
        go.Histogram(
            x=dataframe['MeanFibreDiameter'],
            name='Mean Fibre Diameter'
        ),
        rows=2, cols=1
    )

    fig.add_traces(
        go.Histogram(
            x=dataframe['EstimatedChannelWidth'],
            name='Estimated Channel Width'
            ),
        rows=2, cols=2
    )

    fig.show()

def make3dPlots(dataframe):
    # fig = make_subplots(rows=2, cols=1)
    # fig.add_traces(
    #     go.Histogram(
    #         x=dataframe['noPixels'],
    #         name='number of pixels per component',
    #         marginal='box',
    #         # nbinsx=50,
    #         # xbins=dict(
    #         #     size=50
    #         # )
    #     ),
    #     rows=1, cols=1
    # )
    # fig.add_traces(
    #     go.Histogram(
    #         x=dataframe['volume'],
    #         name='volume of components',
    #         # nbinsx=50,
    #         # xbins=dict(
    #         #     size=50
    #         # )
    #     ),
    #     rows=2, cols=1
    # )

    # fig.show()

    fig = px.histogram(dataframe, x='noPixels', marginal='box')
    fig.show()

    fig2 = px.histogram(dataframe, x='volume', marginal='box')
    fig2.show()

dfX = pd.read_pickle('2DXstatsFULLSTACK.pkl')
dfY = pd.read_pickle('2DYstatsFULLSTACK.pkl')
dfZ = pd.read_pickle('2DZstatsFULLSTACK.pkl')
dfStack = pd.read_pickle('3DstatsFULLSTACK.pkl')

# make2dPlots(dfX)
# make2dPlots(dfY)
make2dPlots(dfZ)

def rejectOutliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
    
def rejectValuesBelowMaxAndAboveMin(data, max, min):
    data = data[data > min]
    return data[data < max]

# dfStack['noPixels'] = rejectOutliers(dfStack['noPixels'], 1)
# dfStack['volume'] = rejectOutliers(dfStack['volume'], 1)

dfStack['noPixels'] = rejectValuesBelowMaxAndAboveMin(dfStack['noPixels'], 50000000, 100)
dfStack['volume'] = rejectValuesBelowMaxAndAboveMin(dfStack['volume'], 100000, 25)


make3dPlots(dfStack)
